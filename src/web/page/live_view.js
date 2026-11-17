// SPDX-License-Identifier: 0BSD
/**
 * @fileoverview live_view.js - the page's 3D view of the live model on three.js
 * 0.187.0: the live model message of the WebSocket (docs/ROUTES.md, "The live
 * model") drawn as colored points, seen from the camera's pose that came with
 * it until the operator turns the view, and from there again when the operator
 * asks or a new scan starts. It draws only when something changed.
 */

import {BufferAttribute, BufferGeometry, LinearSRGBColorSpace, PerspectiveCamera, Points, PointsMaterial, Quaternion, Scene, Vector3, WebGLRenderer} from 'three';

import {OrbitControls} from './three/OrbitControls.js';

/**
 * The live model message's header, in bytes (docs/ROUTES.md, "The live
 * model").
 */
const HEADER_BYTES = 80;

/**
 * Where the header holds has_frame and n, unsigned 32-bit, and the pose, seven
 * doubles from byte 24, 8 bytes each (docs/ROUTES.md, "The live model").
 */
const HAS_FRAME_AT = 16;
const COUNT_AT = 20;
const POSE_AT = 24;
const POSE_VALUES = 7;
const DOUBLE_BYTES = 8;

/**
 * A point's position and color in the message, in bytes: three floats and
 * three bytes.
 */
const POSITION_BYTES = 12;
const COLOR_BYTES = 3;

/**
 * The points' size on the screen, in CSS pixels: 3 [?: a choice]. A size in the
 * model's frame would need the live model's spacing, which the message does not
 * carry, and the live voxel is 2 mm or 8 mm by the object class
 * (src/scan/object_class.cpp).
 */
const POINT_SIZE_PX = 3;

/**
 * The tangents of half the camera's field, across and up, 640 / 612.369 and
 * 400 / 612.286: about 92° by 66° [S: computed from the intrinsics at 1280x800
 * in README.md, "Run", the principal point taken at the image's centre]. The
 * view shows at least the camera's field at any aspect.
 */
const CAMERA_TAN_HALF_ACROSS = 640 / 612.369;
const CAMERA_TAN_HALF_UP = 400 / 612.286;

/**
 * The view's near and far planes, in m: 0.01 and 50 [?: a choice], around the
 * object classes' depth ranges.
 */
const NEAR_M = 0.01;
const FAR_M = 50;

/**
 * The distance of the turning point in front of the camera when the live model
 * holds no point, in m: 0.5 [?: a choice, within the object classes' depth
 * ranges].
 */
const DEFAULT_TARGET_M = 0.5;

/**
 * The least distance of the turning point in front of the camera, in m:
 * 0.1 [?: a choice].
 */
const LEAST_TARGET_M = 0.1;

/**
 * The turn from the scanner's camera frame (x right, y down, z forward) into
 * three.js's camera (x right, y up, looking along -z): half a turn about x. A
 * three.js Quaternion can be changed, so its name is not a constant's; it is
 * only read.
 */
const cameraFrameFlip = new Quaternion(1, 0, 0, 0);

/**
 * A live model as the message holds it.
 * @typedef {{
 *   hasFrame: boolean,
 *   position: !Float32Array,
 *   color: !Uint8Array,
 *   pose: !Array<number>,
 * }}
 */
let LiveModel;

/**
 * Reads a live model message (docs/ROUTES.md, "The live model"): little-endian,
 * a header of 80 bytes, then the positions, then the colors, read in place
 * without a copy.
 * @param {!ArrayBuffer} buffer The message; it must outlive the model given
 *     back.
 * @return {?LiveModel} The model: whether it was taken at a frame, its
 *     positions in m in the models' frame, its colors 0 to 255 and the pose of
 *     its frame (tx, ty, tz in m, then qx, qy, qz, qw from the camera frame
 *     into the models' frame); null when the message is shorter than its header
 *     or its length is not 80 + 15 n bytes for its n points.
 */
function readLiveModel(buffer) {
  if (buffer.byteLength < HEADER_BYTES) {
    return null;
  }
  const header = new DataView(buffer, 0, HEADER_BYTES);
  const hasFrame = header.getUint32(HAS_FRAME_AT, true) === 1;
  const count = header.getUint32(COUNT_AT, true);
  if (buffer.byteLength !==
      HEADER_BYTES + (POSITION_BYTES + COLOR_BYTES) * count) {
    return null;
  }
  const pose = [];
  for (let i = 0; i < POSE_VALUES; i++) {
    pose.push(header.getFloat64(POSE_AT + DOUBLE_BYTES * i, true));
  }
  return {
    hasFrame,
    position: new Float32Array(buffer, HEADER_BYTES, 3 * count),
    color: new Uint8Array(
        buffer, HEADER_BYTES + POSITION_BYTES * count, 3 * count),
    pose,
  };
}

/**
 * Gives the mean of a model's points.
 * @param {!Float32Array} position The points' x, y, z, in m in the models'
 *     frame.
 * @return {?Vector3} The mean, in m in the models' frame; null for no point.
 */
function meanOf(position) {
  const count = position.length / 3;
  if (count === 0) {
    return null;
  }
  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < position.length; i += 3) {
    x += position[i];
    y += position[i + 1];
    z += position[i + 2];
  }
  return new Vector3(x / count, y / count, z / count);
}

/**
 * The 3D view of the live model in an element of the page.
 */
export class LiveView {
  /**
   * Builds the view into its element and draws it empty, following the camera.
   * @param {!HTMLElement} container The element the view fills; the page's,
   *     which keeps it.
   * @param {function(boolean)} onFollowing Called with whether the view follows
   *     the camera, each time that changes.
   * @throws {!Error} When the browser gives no WebGL 2 context (three.js's
   *     renderer throws).
   */
  constructor(container, onFollowing) {
    /** @private @const {!HTMLElement} */
    this.container_ = container;

    /** @private @const {function(boolean)} */
    this.onFollowing_ = onFollowing;

    /** @private @const {!WebGLRenderer} */
    this.renderer_ = new WebGLRenderer({antialias: false});
    // The camera's colors are sRGB bytes; points take no light, so they are
    // written unchanged.
    this.renderer_.outputColorSpace = LinearSRGBColorSpace;
    this.renderer_.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer_.domElement);

    /** @private @const {!Scene} */
    this.scene_ = new Scene();

    /** @private @const {!PerspectiveCamera} */
    this.camera_ = new PerspectiveCamera(60, 1, NEAR_M, FAR_M);
    // Up is -y of the first fused frame's camera, held roughly upright; the
    // controls turn the view about it, so it is set before they are built.
    this.camera_.up.set(0, -1, 0);

    /** @private @const {!Points} */
    this.points_ = new Points(
        new BufferGeometry(),
        new PointsMaterial(
            {size: POINT_SIZE_PX, sizeAttenuation: false, vertexColors: true}));
    // Its bounds are not computed: the view draws every point.
    this.points_.frustumCulled = false;
    this.scene_.add(this.points_);

    /** @private @const {!OrbitControls} */
    this.controls_ = new OrbitControls(this.camera_, this.renderer_.domElement);
    this.controls_.addEventListener('start', () => this.operatorTurns_());
    this.controls_.addEventListener('change', () => this.requestDraw_());

    /** @private {boolean} Whether the view stands at the camera's pose. */
    this.following_ = true;

    /**
     * @private {?Array<number>} The pose of the newest model, as readLiveModel
     *     gives it.
     */
    this.pose_ = null;

    /** @private {?LiveModel} The newest model, not yet drawn. */
    this.pending_ = null;

    /**
     * @private {boolean} Whether a draw is asked for the next animation frame.
     */
    this.drawAsked_ = false;

    new ResizeObserver(() => this.resize_()).observe(container);
    this.resize_();
  }

  /**
   * Takes a live model message; the newest is drawn at the next animation frame
   * and older ones not yet drawn are dropped. The empty model a scan starts
   * with makes the view follow the camera again.
   * @param {!ArrayBuffer} buffer The message; the view keeps it while it draws
   *     its points.
   * @return {boolean} Whether the message was a live model; false leaves the
   *     view as it was.
   */
  show(buffer) {
    const model = readLiveModel(buffer);
    if (model === null) {
      return false;
    }
    this.pending_ = model;
    this.requestDraw_();
    return true;
  }

  /**
   * Makes the view stand at the camera's pose again, the newest model's.
   */
  follow() {
    this.setFollowing_(true);
    this.placeAtPose_();
    this.requestDraw_();
  }

  /**
   * Sets whether the view follows the camera and tells the page when that
   * changes.
   * @param {boolean} following Whether it follows.
   * @private
   */
  setFollowing_(following) {
    if (following !== this.following_) {
      this.following_ = following;
      this.onFollowing_(following);
    }
  }

  /**
   * The controls' start: the operator turns, moves or zooms the view, which
   * stops following. The point the view turns about is put on the view's axis
   * at the model's mean distance, so that the view does not jump.
   * @private
   */
  operatorTurns_() {
    if (!this.following_) {
      return;
    }
    this.setFollowing_(false);
    const forward =
        new Vector3(0, 0, -1).applyQuaternion(this.camera_.quaternion);
    const mean = meanOf(
        this.points_.geometry.getAttribute('position')?.array ??
        new Float32Array());
    const distance = mean === null ?
        DEFAULT_TARGET_M :
        Math.max(LEAST_TARGET_M, mean.sub(this.camera_.position).dot(forward));
    this.controls_.target.copy(this.camera_.position)
        .addScaledVector(forward, distance);
  }

  /**
   * Puts the view's camera at the newest model's pose; nothing before the first
   * model taken at a frame.
   * @private
   */
  placeAtPose_() {
    if (this.pose_ === null) {
      return;
    }
    const [tx, ty, tz, qx, qy, qz, qw] = this.pose_;
    this.camera_.position.set(tx, ty, tz);
    this.camera_.quaternion.set(qx, qy, qz, qw).multiply(cameraFrameFlip);
  }

  /**
   * Fits the drawing to the element's size and the field to the camera's: at
   * least the camera's across and up, the rest of the element's aspect added.
   * @private
   */
  resize_() {
    const width = Math.max(1, this.container_.clientWidth);
    const height = Math.max(1, this.container_.clientHeight);
    this.renderer_.setSize(width, height, false);
    const aspect = width / height;
    const tanHalfUp =
        Math.max(CAMERA_TAN_HALF_UP, CAMERA_TAN_HALF_ACROSS / aspect);
    this.camera_.aspect = aspect;
    this.camera_.fov = 2 * Math.atan(tanHalfUp) * 180 / Math.PI;
    this.camera_.updateProjectionMatrix();
    this.requestDraw_();
  }

  /**
   * Asks for a draw at the next animation frame, once however often it is
   * asked.
   * @private
   */
  requestDraw_() {
    if (this.drawAsked_) {
      return;
    }
    this.drawAsked_ = true;
    window.requestAnimationFrame(() => this.draw_());
  }

  /**
   * Draws the view: the newest model first replaces the points, its pose places
   * the camera while the view follows, and the empty model of a scan's start
   * makes it follow again.
   * @private
   */
  draw_() {
    this.drawAsked_ = false;
    const model = this.pending_;
    if (model !== null) {
      this.pending_ = null;
      const geometry = new BufferGeometry();
      geometry.setAttribute('position', new BufferAttribute(model.position, 3));
      geometry.setAttribute('color', new BufferAttribute(model.color, 3, true));
      this.points_.geometry.dispose();
      this.points_.geometry = geometry;
      if (model.hasFrame) {
        this.pose_ = model.pose;
      } else {
        this.pose_ = null;
        this.setFollowing_(true);
      }
      if (this.following_) {
        this.placeAtPose_();
      }
    }
    this.renderer_.render(this.scene_, this.camera_);
  }
}
