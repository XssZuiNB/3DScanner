// SPDX-License-Identifier: 0BSD
/**
 * @fileoverview scan_page.js - what the page on which an operator makes a scan
 * does (docs/ROUTES.md; the spec, "The web layer and the page"): it keeps a
 * WebSocket to the scanner open, reconnecting when it closes, shows each state
 * it pushes in the panel of the state's phase with what the operator can do
 * next, draws each live model in the 3D view (live_view.js), and gives the
 * operator's commands, start, stop and discard, each refusal shown in plain
 * words in the page that gave the command until its next command or until a
 * scan scans. The discard asks once. The "too close" notice is smoothed here
 * against flicker (R36). Every name of a route and of a message is
 * docs/ROUTES.md's.
 */

import {LiveView} from './live_view.js';

/**
 * The WebSocket that pushes the state and the live model (docs/ROUTES.md,
 * "Routes").
 */
const LIVE_ROUTE = '/api/live';

/** The commands' routes, each a POST with its argument in the request line. */
const START_ROUTE = '/api/start?object_class=';
const STOP_ROUTE = '/api/stop';
const DISCARD_ROUTE = '/api/discard';

/**
 * The wait before the first reconnection, in ms, doubled after each one that
 * fails, up to the last: 1 s to 8 s [?: a choice]. A page brought back to the
 * screen reconnects at once.
 */
const RECONNECT_FIRST_MS = 1000;
const RECONNECT_LAST_MS = 8000;

/**
 * The time constant of the "too close" notice's average over time, in s:
 * 0.3 [?: a choice]. The state's too_close is one frame's (R36), a state about
 * each frame [S: docs/ROUTES.md, "The WebSocket"], and flickers at the rule's
 * threshold.
 */
const TOO_CLOSE_TIME_S = 0.3;

/**
 * The average at which the notice shows and at which it hides: 0.6 and
 * 0.3 [?: a choice], apart, so that an average near one of them does not make
 * the notice flicker.
 */
const TOO_CLOSE_SHOW = 0.6;
const TOO_CLOSE_HIDE = 0.3;

/**
 * What the tracking check's failed condition means to the operator
 * (docs/methods/tracking.md, "The check"), by the state's lost_cause.
 * @const {!Object<string, string>}
 */
const LOST_CAUSES = {
  'registration_end': 'The view could not be aligned with the model.',
  'fitness': 'Too little of the view matches the model.',
  'inlier_rmse': 'The view does not fit the model closely: did something move?',
  'step': 'The camera moved too fast.',
};

/**
 * Why a scan ended, by the state's end, one of docs/SCAN_FILES.md's closed set;
 * the state's end_message, empty for the operator's stop and the service's,
 * goes after it.
 * @const {!Object<string, string>}
 */
const ENDS = {
  'stopped': 'The operator stopped it',
  'source_timeout': 'It ended by itself: the camera gave no frame in time',
  'source_failure': 'It ended by itself: the camera failed',
  'source_end': 'It ended by itself: the frames ran out',
  'tracking_failed': 'It ended by itself: tracking failed',
  'writer_failed': 'It ended by itself: its frames could not be written',
  'shut_down': 'The scanner\'s service stopped during it',
};

/**
 * The scanner's state as the WebSocket pushes it (docs/ROUTES.md, "The state").
 * @typedef {{
 *   phase: string,
 *   scan: string,
 *   tracking: string,
 *   lost_cause: string,
 *   too_close: boolean,
 *   frames: {fused: number, lost: number},
 *   end: string,
 *   end_message: string,
 *   outcome: string,
 *   outcome_message: string,
 *   live_model_failure: string,
 *   property_refusals: string,
 * }}
 */
let ScanState;

/**
 * The "too close" notice's smoothing: an average of the state's too_close over
 * time, which shows the notice above one level and hides it below a lower one.
 */
class TooCloseNotice {
  /** Builds it hidden. */
  constructor() {
    /** @private {number} The average, 0 to 1. */
    this.average_ = 0;

    /**
     * @private {?number} When the last state came, in s; null before the
     *     first.
     */
    this.lastS_ = null;

    /** @private {boolean} Whether the notice shows. */
    this.shown_ = false;
  }

  /**
   * Takes one state's too_close.
   * @param {boolean} tooClose Whether the state's last frame was too close.
   * @param {number} nowS When the state came, in s, on the page's monotonic
   *     clock.
   * @return {boolean} Whether the notice shows.
   */
  update(tooClose, nowS) {
    const elapsedS = this.lastS_ === null ? 0 : Math.max(0, nowS - this.lastS_);
    this.lastS_ = nowS;
    const weight = 1 - Math.exp(-elapsedS / TOO_CLOSE_TIME_S);
    this.average_ += ((tooClose ? 1 : 0) - this.average_) * weight;
    if (this.average_ >= TOO_CLOSE_SHOW) {
      this.shown_ = true;
    } else if (this.average_ <= TOO_CLOSE_HIDE) {
      this.shown_ = false;
    }
    return this.shown_;
  }

  /** Hides the notice and forgets the average, as outside a scan that scans. */
  reset() {
    this.average_ = 0;
    this.lastS_ = null;
    this.shown_ = false;
  }
}

/**
 * The WebSocket to the scanner, opened again after it closes.
 */
class Connection {
  /**
   * Builds the connection, closed; open() opens it.
   * @param {function(string)} onState Called with each state message's text.
   * @param {function(!ArrayBuffer)} onLiveModel Called with each live model
   *     message.
   * @param {function(boolean)} onOpen Called with whether the connection is
   *     open, each time that changes.
   */
  constructor(onState, onLiveModel, onOpen) {
    /** @private @const {function(string)} */
    this.onState_ = onState;

    /** @private @const {function(!ArrayBuffer)} */
    this.onLiveModel_ = onLiveModel;

    /** @private @const {function(boolean)} */
    this.onOpen_ = onOpen;

    /** @private {?WebSocket} The socket while it connects or is open. */
    this.socket_ = null;

    /** @private {?number} The timer of the next reconnection. */
    this.timer_ = null;

    /** @private {number} The wait before the next reconnection, in ms. */
    this.waitMs_ = RECONNECT_FIRST_MS;

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible' && this.socket_ === null) {
        this.open();
      }
    });
  }

  /**
   * Opens the WebSocket to the page's own host, unless one is open or
   * connecting.
   */
  open() {
    if (this.timer_ !== null) {
      window.clearTimeout(this.timer_);
      this.timer_ = null;
    }
    if (this.socket_ !== null) {
      return;
    }
    // The page's own host and port, the scheme the page came with turned into
    // the WebSocket's.
    const url = new URL(LIVE_ROUTE, window.location.href);
    url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(url.href);
    socket.binaryType = 'arraybuffer';
    socket.addEventListener('open', () => {
      this.waitMs_ = RECONNECT_FIRST_MS;
      this.onOpen_(true);
    });
    socket.addEventListener('message', (event) => {
      if (typeof event.data === 'string') {
        this.onState_(event.data);
      } else {
        this.onLiveModel_(/** @type {!ArrayBuffer} */ (event.data));
      }
    });
    socket.addEventListener('close', () => {
      this.socket_ = null;
      this.onOpen_(false);
      this.timer_ = window.setTimeout(() => this.open(), this.waitMs_);
      this.waitMs_ = Math.min(2 * this.waitMs_, RECONNECT_LAST_MS);
    });
    this.socket_ = socket;
  }
}

/**
 * Gives a scan's directory name as a time to read: 20261117T101112.123456Z as
 * 2026-11-17 10:11:12 UTC (docs/SCAN_FILES.md, "A scan's directory").
 * @param {string} scan The name.
 * @return {string} The time, or the name as it is when it has another form.
 */
function startOf(scan) {
  const parts = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})/.exec(scan);
  if (parts === null) {
    return scan;
  }
  const [, year, month, day, hour, minute, second] = parts;
  return `${year}-${month}-${day} ${hour}:${minute}:${second} UTC`;
}

/**
 * Gives why a scan ended, in plain words.
 * @param {!ScanState} state The state.
 * @return {string} The reason, a sentence; empty before the scan ended and for
 *     a scan discarded.
 */
function endOf(state) {
  if (state.end === '') {
    return '';
  }
  const words = ENDS[state.end] ?? `It ended: ${state.end}`;
  return state.end_message === '' ? `${words}.` :
                                    `${words} (${state.end_message}).`;
}

/**
 * Gives whether a phase is one of a scan in progress, whose live model the
 * view shows and which the operator may discard.
 * @param {string} phase The state's phase.
 * @return {boolean} Whether it is scanning or refining.
 */
function scanInProgress(phase) {
  return phase === 'scanning' || phase === 'refining';
}

/**
 * Gives a command's refusal from the scanner's answer (docs/ROUTES.md, "A
 * command's answer").
 * @param {!Response} response The answer.
 * @return {!Promise<string>} The refusal in the scanner's words; empty when the
 *     command was taken; that the answer could not be read, with its HTTP
 *     status, when it is no answer.
 */
async function refusalOf(response) {
  let answer;
  try {
    answer = await response.json();
  } catch (error) {
    return `The scanner's answer could not be read (HTTP ${response.status}).`;
  }
  return answer.accepted === true ? '' :
                                    `The scanner refused: ${answer.refusal}`;
}

/**
 * Gives what became of the last scan, in plain words, for the idle panel.
 * @param {!ScanState} state The state.
 * @return {string} The outcome with why the scan ended; empty before the first
 *     scan ended.
 */
function lastScanOf(state) {
  const scan = `The last scan, started ${startOf(state.scan)},`;
  const ended = endOf(state);
  switch (state.outcome) {
    case 'refined':
      return state.outcome_message === '' ?
          `${scan} is refined and kept. ${ended}` :
          `${scan} is refined and kept; ${state.outcome_message}. ${ended}`;
    case 'unfinished':
      return `${scan} is unfinished: ${state.outcome_message}. ${ended}`;
    case 'discarded':
      return `${scan} is discarded: nothing of it stays on disk.`;
    default:
      return '';
  }
}

/**
 * The page: its parts, the state shown and the commands.
 */
class ScanPage {
  /**
   * Finds the page's parts and wires the operator's buttons; nothing shows
   * until open().
   */
  constructor() {
    /** @private @const {!HTMLElement} */
    this.connectionBanner_ = this.part_('connection');
    /** @private @const {!HTMLElement} */
    this.tracking_ = this.part_('tracking');
    /** @private @const {!HTMLElement} */
    this.trackingWord_ = this.part_('tracking-word');
    /** @private @const {!HTMLElement} */
    this.trackingDetail_ = this.part_('tracking-detail');
    /** @private @const {!HTMLElement} */
    this.tooClose_ = this.part_('too-close');
    /** @private @const {!HTMLElement} */
    this.follow_ = this.part_('follow');
    /** @private @const {!HTMLElement} */
    this.lastScan_ = this.part_('last-scan');
    /** @private @const {!HTMLElement} */
    this.counts_ = this.part_('counts');
    /** @private @const {!HTMLElement} */
    this.refiningEnd_ = this.part_('refining-end');
    /** @private @const {!HTMLElement} */
    this.liveModelFailure_ = this.part_('live-model-failure');
    /** @private @const {!HTMLElement} */
    this.propertyRefusals_ = this.part_('property-refusals');
    /** @private @const {!HTMLElement} */
    this.refusal_ = this.part_('refusal');
    /** @private @const {!HTMLButtonElement} */
    this.start_ = /** @type {!HTMLButtonElement} */ (this.part_('start'));
    /** @private @const {!HTMLButtonElement} */
    this.stop_ = /** @type {!HTMLButtonElement} */ (this.part_('stop'));
    /** @private @const {!HTMLButtonElement} */
    this.discard_ = /** @type {!HTMLButtonElement} */ (this.part_('discard'));
    /** @private @const {!HTMLDialogElement} */
    this.discardDialog_ =
        /** @type {!HTMLDialogElement} */ (this.part_('discard-dialog'));
    /** @private @const {!NodeList<!HTMLElement>} */
    this.phases_ = document.querySelectorAll('[data-phase]');
    /** @private @const {!NodeList<!HTMLInputElement>} */
    this.classes_ = document.querySelectorAll('input[name="object-class"]');

    /**
     * @private {?LiveView} The 3D view; null when the browser cannot draw it.
     */
    this.view_ = this.buildView_();

    /** @private @const {!TooCloseNotice} */
    this.notice_ = new TooCloseNotice();

    /** @private @const {!Connection} */
    this.connection_ = new Connection(
        (text) => this.takeState_(text),
        (buffer) => this.takeLiveModel_(buffer),
        (open) => this.takeOpen_(open));

    /** @private {?ScanState} The newest state; null before the first. */
    this.state_ = null;

    /** @private {boolean} Whether the WebSocket is open. */
    this.connected_ = false;

    /**
     * @private {boolean} Whether a command of this page waits for its answer.
     */
    this.busy_ = false;

    /**
     * @private {boolean} Whether the next change into idle keeps the refusal:
     *     a start of this page refused while the page still showed it starting,
     *     whose idle state comes after the answer.
     */
    this.keepAtIdle_ = false;

    /**
     * @private {?string} The scan the open discard dialog is for; null while
     *     it is closed.
     */
    this.discardScan_ = null;

    /**
     * @private {boolean} Whether the WebSocket was open before, for the
     *     banner's words.
     */
    this.wasOpen_ = false;

    this.start_.addEventListener('click', () => this.startScan_());
    this.stop_.addEventListener('click', () => this.command_(STOP_ROUTE));
    this.discard_.addEventListener('click', () => {
      // A dialog closed by Escape keeps its last returnValue, which may be the
      // last discard's.
      this.discardDialog_.returnValue = '';
      this.discardScan_ = this.state_?.scan ?? null;
      this.discardDialog_.showModal();
    });
    this.discardDialog_.addEventListener('close', () => {
      this.discardScan_ = null;
      if (this.discardDialog_.returnValue === 'discard') {
        this.command_(DISCARD_ROUTE);
      }
    });
    this.follow_.addEventListener('click', () => this.view_?.follow());
    for (const choice of this.classes_) {
      choice.addEventListener('change', () => this.showButtons_());
    }
  }

  /** Connects to the scanner. */
  open() {
    this.connection_.open();
  }

  /**
   * Finds a part of the page by its id.
   * @param {string} id The id, index.html's.
   * @return {!HTMLElement} The part.
   * @throws {!Error} When index.html has no such part.
   * @private
   */
  part_(id) {
    const part = document.getElementById(id);
    if (part === null) {
      throw new Error(`index.html has no part "${id}"`);
    }
    return part;
  }

  /**
   * Builds the 3D view, or says why the browser cannot draw it; the rest of the
   * page works without it.
   * @return {?LiveView} The view; null when the browser gives no WebGL 2.
   * @private
   */
  buildView_() {
    try {
      return new LiveView(this.part_('view'), (following) => {
        this.follow_.hidden = following;
      });
    } catch (error) {
      const failure = this.part_('view-failure');
      failure.textContent =
          `This browser cannot draw the live model: ${error.message}`;
      failure.hidden = false;
      return null;
    }
  }

  /**
   * Takes a state message and shows it.
   * @param {string} text The message, docs/ROUTES.md's state.
   * @private
   */
  takeState_(text) {
    let state;
    try {
      state = /** @type {!ScanState} */ (JSON.parse(text));
    } catch (error) {
      console.warn(`A state message could not be read: ${error.message}`);
      return;
    }
    const phaseChanged =
        this.state_ === null || this.state_.phase !== state.phase;
    this.state_ = state;
    // A refusal goes with the phase it was given in, but not while a command
    // waits for its answer, which shows its own, nor at the idle state of a
    // start that failed, which the scanner publishes before it answers and
    // which may reach the page after the answer.
    if (phaseChanged && !this.busy_) {
      if (!(state.phase === 'idle' && this.keepAtIdle_)) {
        this.refusal_.hidden = true;
      }
      this.keepAtIdle_ = false;
    }
    // The dialog asks about one scan; it never discards another.
    if (this.discardScan_ !== null &&
        (!scanInProgress(state.phase) || state.scan !== this.discardScan_)) {
      this.discardDialog_.close('keep');
    }
    this.showState_(state);
  }

  /**
   * Takes a live model message and draws it.
   * @param {!ArrayBuffer} buffer The message, docs/ROUTES.md's live model.
   * @private
   */
  takeLiveModel_(buffer) {
    if (this.view_ !== null && !this.view_.show(buffer)) {
      const bytes = buffer.byteLength;
      console.warn(`A live model message of ${bytes} bytes could not be read.`);
    }
  }

  /**
   * Takes whether the WebSocket is open and shows it.
   * @param {boolean} open Whether it is open.
   * @private
   */
  takeOpen_(open) {
    this.connected_ = open;
    if (open) {
      this.wasOpen_ = true;
    } else if (this.discardScan_ !== null) {
      this.discardDialog_.close('keep');
    }
    this.connectionBanner_.textContent = this.wasOpen_ ?
        'The connection to the scanner is lost. Reconnecting…' :
        'Connecting to the scanner…';
    this.connectionBanner_.hidden = open;
    if (this.state_ !== null) {
      this.showState_(this.state_);
    }
  }

  /**
   * Shows a state: the panel of its phase, the tracking state, the notice, the
   * live model that could not be taken and the camera's refused settings, and
   * what the last scan became.
   * @param {!ScanState} state The state.
   * @private
   */
  showState_(state) {
    for (const panel of this.phases_) {
      panel.hidden = panel.dataset.phase !== state.phase;
    }
    document.body.dataset.view = scanInProgress(state.phase) ? 'on' : 'off';
    const scanning = state.phase === 'scanning';
    this.showTracking_(state);
    // Without the connection the last state is old: no notice stands on it.
    if (scanning && this.connected_) {
      this.tooClose_.hidden =
          !this.notice_.update(state.too_close, performance.now() / 1000);
    } else {
      this.notice_.reset();
      this.tooClose_.hidden = true;
    }
    const frames = state.frames;
    this.counts_.textContent =
        `Frames: ${frames.fused} in the model, ${frames.lost} lost.`;
    const lastScan = lastScanOf(state);
    this.lastScan_.textContent = lastScan;
    this.lastScan_.hidden = lastScan === '';
    const ended = state.end === 'stopped' ? '' : endOf(state);
    this.refiningEnd_.textContent = ended;
    this.refiningEnd_.hidden = ended === '';
    const failure = state.live_model_failure;
    this.liveModelFailure_.textContent =
        `A live model could not be taken: ${failure}. The scan goes on.`;
    this.liveModelFailure_.hidden =
        failure === '' || !scanInProgress(state.phase);
    const refusals = state.property_refusals;
    this.propertyRefusals_.textContent = `${refusals}. The scan goes on.`;
    this.propertyRefusals_.hidden =
        refusals === '' || !scanInProgress(state.phase);
    this.showButtons_();
  }

  /**
   * Shows the tracking state while a scan scans: what tracking made of the last
   * frame, and on a tracking loss what the operator does and why it was lost;
   * unknown while the connection is lost.
   * @param {!ScanState} state The state.
   * @private
   */
  showTracking_(state) {
    this.tracking_.hidden = state.phase !== 'scanning';
    // Without the connection the last state is old, so tracking is unknown.
    const tracking = this.connected_ ? state.tracking : 'unknown';
    this.tracking_.classList.toggle('is-tracking', tracking === 'tracking');
    this.tracking_.classList.toggle('is-lost', tracking === 'lost');
    this.tracking_.classList.toggle('is-failed', tracking === 'failed');
    switch (tracking) {
      case 'unknown':
        this.trackingWord_.textContent = 'Tracking unknown';
        this.trackingDetail_.textContent =
            'The connection to the scanner is lost.';
        break;
      case 'tracking':
        this.trackingWord_.textContent = 'Tracking';
        this.trackingDetail_.textContent = 'The live model grows.';
        break;
      case 'lost':
        this.trackingWord_.textContent = 'Tracking lost';
        this.trackingDetail_.textContent =
            'Point the camera back at a part already scanned. ' +
            (LOST_CAUSES[state.lost_cause] ?? '');
        break;
      case 'failed':
        this.trackingWord_.textContent = 'Tracking failed';
        this.trackingDetail_.textContent = 'The scan ends.';
        break;
      default:
        this.trackingWord_.textContent = 'Waiting for the first frame';
        this.trackingDetail_.textContent = '';
    }
  }

  /**
   * Shows the buttons the phase offers, each usable while the WebSocket is open
   * and no command of this page waits for its answer.
   * @private
   */
  showButtons_() {
    const phase = this.state_?.phase ?? '';
    const ready = this.connected_ && !this.busy_;
    this.start_.hidden = phase !== 'idle';
    this.start_.disabled = !ready || this.chosenClass_() === '';
    this.stop_.hidden = phase !== 'scanning';
    this.stop_.disabled = !ready;
    this.discard_.hidden = !scanInProgress(phase);
    this.discard_.disabled = !ready;
  }

  /**
   * Gives the object class the operator chose.
   * @return {string} Its name, docs/ROUTES.md's; empty before a choice.
   * @private
   */
  chosenClass_() {
    for (const choice of this.classes_) {
      if (choice.checked) {
        return choice.value;
      }
    }
    return '';
  }

  /**
   * Starts a scan of the chosen object class.
   * @private
   */
  startScan_() {
    this.command_(START_ROUTE + encodeURIComponent(this.chosenClass_()));
  }

  /**
   * Gives a command and shows its refusal: the scanner's in its own words, or
   * that it did not answer. The command's buttons wait meanwhile; a start waits
   * for the camera and the model's memory.
   * @param {string} route The command's route with its argument.
   * @return {!Promise<void>} Settled once the answer is shown; it never
   *     rejects.
   * @private
   */
  async command_(route) {
    this.busy_ = true;
    this.keepAtIdle_ = false;
    this.refusal_.hidden = true;
    this.showButtons_();
    let refusal;
    try {
      refusal = await refusalOf(await fetch(route, {method: 'POST'}));
    } catch (error) {
      refusal = `The scanner did not answer: ${error.message}`;
    }
    this.refusal_.textContent = refusal;
    this.refusal_.hidden = refusal === '';
    this.keepAtIdle_ = refusal !== '' && route.startsWith(START_ROUTE) &&
        this.state_?.phase === 'starting';
    this.busy_ = false;
    this.showButtons_();
  }
}

new ScanPage().open();
