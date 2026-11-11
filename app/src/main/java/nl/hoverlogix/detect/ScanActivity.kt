package nl.hoverlogix.detect

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.RectF
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.DisplayOrientedMeteringPointFactory
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.CompletableFuture
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * Guided scan of one pallet:
 *  1. SEARCH   live detection until a pallet label and a location label are both in view
 *  2. CAPTURING zoom and focus on the location labels matched to that pallet, take a still, read them;
 *               then zoom and focus on the pallet label, take a still, read barcodes and text
 *  3. RESULT   the user confirms (or picks) the location and saves; crops + scan.json go to ScanStore
 *
 * The camera steps (zoomTo, grabStill) are the only device-specific part. On a drone the same flow
 * runs with the gimbal/zoom camera behind those two functions.
 */
class ScanActivity : AppCompatActivity() {

    private enum class State { SEARCH, CAPTURING, AIMING, RESULT }

    /** One back camera. [magnification] is relative to the main lens (3.0 = a 3x telephoto). */
    private class Lens(val info: CameraInfo, val id: String, val magnification: Float)

    companion object {
        /** Photos per label; the sharpest one that decodes is used. */
        const val BURST = 4
        /** Smallest pallet-label width in photo pixels at which barcodes are normally readable. */
        const val MIN_LABEL_PX = 300
        /** Highest total zoom (lens magnification x digital zoom) used for a label photo. */
        const val MAX_ZOOM = 8f
        /** Share of the picture width a label should fill after zooming. */
        const val PALLET_LABEL_FILL = 0.4f
        const val LOCATION_LABEL_FILL = 0.25f
    }

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private lateinit var autoScan: CheckBox
    private lateinit var btnScan: Button
    private lateinit var resultPanel: View
    private lateinit var candidatesRow: LinearLayout
    private lateinit var locationHint: TextView
    private lateinit var locationId: EditText
    private lateinit var palletImage: ImageView
    private lateinit var palletCodes: TextView
    private lateinit var palletText: TextView
    private lateinit var diagText: TextView

    private lateinit var analysisExecutor: ExecutorService
    private lateinit var scanExecutor: ExecutorService
    private lateinit var captureExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null

    @Volatile private var state = State.SEARCH
    @Volatile private var confidence = 0.35f
    @Volatile private var auto = true
    @Volatile private var lastDets: List<Detection> = emptyList()
    @Volatile private var lastW = 1
    @Volatile private var lastH = 1
    @Volatile private var frameRequest: CompletableFuture<Bitmap>? = null
    @Volatile private var cooldownUntil = 0L
    private var stableFrames = 0

    private var provider: ProcessCameraProvider? = null
    private var lenses: List<Lens> = emptyList()
    @Volatile private var currentLens: Lens? = null
    @Volatile private var aimClass = -1
    @Volatile private var aimTrigger: CompletableFuture<Boolean>? = null
    private var aimStable = 0

    // Result being shown
    private var locations: List<LocationRead> = emptyList()
    private var selected = -1
    private var palletCrop: Bitmap? = null
    private var palletRead: LabelRead? = null
    private var zoomLocation = 1f
    private var zoomPallet = 1f
    private var diagnostics: JSONObject? = null

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, "Camera permission is needed to scan", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Labels.load(this)
        setContentView(R.layout.activity_scan)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        status = findViewById(R.id.status)
        autoScan = findViewById(R.id.autoScan)
        btnScan = findViewById(R.id.btnScan)
        resultPanel = findViewById(R.id.resultPanel)
        candidatesRow = findViewById(R.id.candidates)
        locationHint = findViewById(R.id.locationHint)
        locationId = findViewById(R.id.locationId)
        palletImage = findViewById(R.id.palletImage)
        palletCodes = findViewById(R.id.palletCodes)
        palletText = findViewById(R.id.palletText)
        diagText = findViewById(R.id.diagText)

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        overlay.scaleMode = OverlayView.ScaleMode.FILL
        analysisExecutor = Executors.newSingleThreadExecutor()
        scanExecutor = Executors.newSingleThreadExecutor()
        captureExecutor = Executors.newSingleThreadExecutor()
        confidence = Settings.confidence(this)

        autoScan.setOnCheckedChangeListener { _, on -> auto = on; stableFrames = 0 }
        btnScan.setOnClickListener { if (state == State.AIMING) aimTrigger?.complete(true) else startScan(null) }
        findViewById<Button>(R.id.btnRetake).setOnClickListener { backToSearch(1000) }
        findViewById<Button>(R.id.btnSave).setOnClickListener { saveResult() }

        // Tap a pallet label to scan that one instead of the one nearest the middle.
        overlay.setOnTouchListener { v, e ->
            if (e.action == MotionEvent.ACTION_UP && state == State.SEARCH) {
                val p = overlay.viewToSource(e.x, e.y)
                val pad = 0.03f * lastW
                val hit = lastDets.filter { it.cls == ScanLogic.PALLET_LABEL }.firstOrNull {
                    p.x >= it.box.left - pad && p.x <= it.box.right + pad && p.y >= it.box.top - pad && p.y <= it.box.bottom + pad
                }
                if (hit != null) startScan(hit)
                v.performClick()
            }
            true
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestCamera.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val p = providerFuture.get()
            provider = p
            lenses = findLenses(p)
            try {
                bind(lenses.firstOrNull())
            } catch (e: Exception) {
                status.text = "The camera could not be started: ${e.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * The back cameras this phone lets apps use, main lens first. Magnification comes from focal
     * length and sensor size. Some phones hide their telephoto lenses from apps; then only the main
     * lens is listed and zoom stays digital.
     */
    @androidx.annotation.OptIn(markerClass = [ExperimentalCamera2Interop::class])
    private fun findLenses(p: ProcessCameraProvider): List<Lens> = runCatching {
        val raw = p.availableCameraInfos.mapNotNull { info ->
            val c2 = Camera2CameraInfo.from(info)
            if (c2.getCameraCharacteristic(CameraCharacteristics.LENS_FACING) != CameraMetadata.LENS_FACING_BACK) return@mapNotNull null
            val focal = c2.getCameraCharacteristic(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)?.firstOrNull() ?: return@mapNotNull null
            val size = c2.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE) ?: return@mapNotNull null
            val diagonal = Math.hypot(size.width.toDouble(), size.height.toDouble()).toFloat()
            if (diagonal <= 0f) return@mapNotNull null
            Triple(info, c2.cameraId, focal / diagonal)
        }
        val main = raw.firstOrNull()?.third ?: return@runCatching emptyList<Lens>()
        raw.map { Lens(it.first, it.second, it.third / main) }
    }.getOrDefault(emptyList())

    /** The lens with the most real magnification that does not exceed the wanted total zoom. */
    private fun planLens(zoom: Float): Lens? =
        lenses.filter { it.magnification >= 0.9f && it.magnification <= zoom * 1.15f }.maxByOrNull { it.magnification }
            ?: lenses.firstOrNull()

    /** Binds preview, detection and photo capture to [lens] (null = the default back camera). Main thread. */
    private fun bind(lens: Lens?) {
        val p = provider ?: return
        val selector = lens?.info?.cameraSelector ?: CameraSelector.DEFAULT_BACK_CAMERA
        val selector16x9 = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(
                ResolutionStrategy(Size(1280, 720), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
            )
            .build()
        val preview = Preview.Builder().setResolutionSelector(selector16x9).build()
            .also { it.setSurfaceProvider(previewView.surfaceProvider) }
        val analysis = ImageAnalysis.Builder()
            .setResolutionSelector(selector16x9)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
            .also { it.setAnalyzer(analysisExecutor, ::analyze) }
        // Photos for the label crops. First choice: the sensor's highest resolution (slower, but more
        // real pixels on the label). Then the normal maximum, then no photos.
        fun captureCase(highRes: Boolean) = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                    .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                    .apply { if (highRes) setAllowedResolutionMode(ResolutionSelector.PREFER_HIGHER_RESOLUTION_OVER_CAPTURE_RATE) }
                    .build()
            )
            .build()
        camera = null
        imageCapture = null
        for (highRes in listOf(true, false)) {
            val capture = captureCase(highRes)
            try {
                p.unbindAll()
                camera = p.bindToLifecycle(this, selector, preview, analysis, capture)
                imageCapture = capture
                break
            } catch (e: Exception) {
                // try the next option
            }
        }
        if (camera == null) {
            p.unbindAll()
            camera = p.bindToLifecycle(this, selector, preview, analysis)
        }
        currentLens = lens
    }

    /** Switches lens from the scan thread and waits until the new camera delivers pictures. */
    private fun switchLens(lens: Lens?): Boolean {
        if (lens?.id == currentLens?.id) return true
        val done = CompletableFuture<Boolean>()
        runOnUiThread { done.complete(runCatching { bind(lens) }.isSuccess) }
        val ok = runCatching { done.get(6, TimeUnit.SECONDS) }.getOrDefault(false)
        if (ok) Thread.sleep(500)
        return ok
    }

    // ---------------------------------------------------------------- 1. search

    private fun analyze(image: ImageProxy) {
        try {
            val request = frameRequest
            val st = state
            if (st != State.SEARCH && st != State.AIMING && request == null) return
            val bmp = LiveActivity.rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            if (request != null) {           // a still was asked for and ImageCapture is not available
                frameRequest = null
                request.complete(bmp)
                return
            }
            val dets = detect(bmp)
            lastDets = dets; lastW = bmp.width; lastH = bmp.height
            if (st == State.AIMING) { aim(dets, bmp.width, bmp.height); return }
            val labels = dets.filter { it.cls == ScanLogic.PALLET_LABEL }
            val hasLocation = dets.any { it.cls == ScanLogic.LOCATION_LABEL }
            val target = ScanLogic.centreMost(labels, bmp.width, bmp.height)
            val cands = if (target != null) ScanLogic.locationCandidates(target, dets) else emptyList()

            // A telephoto lens multiplies the real pixels; digital zoom does not.
            val optical = (lenses.maxOfOrNull { it.magnification } ?: 1f).coerceIn(1f, MAX_ZOOM)
            val labelPx = if (target != null) (target.box.width() / bmp.width * photoWidth(bmp.width < bmp.height) * optical).toInt() else 0
            val closeEnough = labelPx == 0 || labelPx >= MIN_LABEL_PX
            stableFrames = if (target != null && hasLocation && closeEnough) stableFrames + 1 else 0
            val matching = cands.count { it.belongs }
            val text = when {
                target == null && !hasLocation -> "Point at a pallet so its label and the location label on the beam are in view"
                target == null -> "Location label found. Bring the pallet label into view"
                !closeEnough -> "Move closer: the pallet label would be only $labelPx px wide in the photo ($MIN_LABEL_PX needed for a sharp read)"
                !hasLocation -> "Pallet label found. Bring the location label on the beam into view"
                else -> "Pallet label and ${if (matching == 0) "no matching" else "$matching matching"} location label${if (matching == 1) "" else "s"}. Hold steady"
            }
            runOnUiThread {
                if (state == State.SEARCH) {
                    overlay.setDetections(dets, bmp.width, bmp.height)
                    overlay.setMarks(target?.box, cands.filter { it.belongs }.map { it.box })
                    status.text = text
                }
            }
            if (auto && stableFrames >= 4 && SystemClock.elapsedRealtime() > cooldownUntil) startScan(null)
        } catch (e: Exception) {
            runOnUiThread { status.text = "Detection error: ${e.message}" }
        } finally {
            image.close()
        }
    }

    /** While zoomed in: guides the user onto the label and fires the burst once it is centred and steady. */
    private fun aim(dets: List<Detection>, w: Int, h: Int) {
        val t = ScanLogic.centreMost(dets.filter { it.cls == aimClass }, w, h)
        val aimed = t != null && t.box.left > 0.02f * w && t.box.right < 0.98f * w && t.box.top > 0.02f * h && t.box.bottom < 0.98f * h &&
            Math.abs(t.box.centerX() - w / 2f) < 0.3f * w && Math.abs(t.box.centerY() - h / 2f) < 0.3f * h
        aimStable = if (aimed) aimStable + 1 else 0
        val what = if (aimClass == ScanLogic.LOCATION_LABEL) "location label on the beam" else "pallet label"
        val text = when {
            t == null -> "Zoomed in. Aim the frame at the $what, or press Take photo now"
            !aimed -> "Bring the $what into the middle of the frame"
            else -> "Hold steady…"
        }
        runOnUiThread {
            if (state == State.AIMING) {
                overlay.setDetections(dets, w, h)
                overlay.setMarks(t?.box, emptyList())
                status.text = text
            }
        }
        if (aimStable >= 3) aimTrigger?.complete(true)
    }

    /** Width in pixels of an upright photo from the capture stream (falls back to the preview frame size). */
    private fun photoWidth(portrait: Boolean): Int {
        val r = imageCapture?.resolutionInfo?.resolution ?: return lastW
        return if (portrait) minOf(r.width, r.height) else maxOf(r.width, r.height)
    }

    private fun detect(bmp: Bitmap): List<Detection> {
        val detector = Detectors.get(this, Detectors.Kind.ACCURATE)
        return synchronized(detector) { detector.detect(bmp, confidence) }
    }

    // ---------------------------------------------------------------- 2. zoom, burst, read

    @Synchronized
    private fun startScan(tapped: Detection?) {
        if (state != State.SEARCH) return
        val dets = lastDets
        val w = lastW
        val h = lastH
        val target = tapped ?: ScanLogic.centreMost(dets.filter { it.cls == ScanLogic.PALLET_LABEL }, w, h)
        if (target == null) {
            runOnUiThread { Toast.makeText(this, "No pallet label in view yet", Toast.LENGTH_SHORT).show() }
            return
        }
        state = State.CAPTURING
        stableFrames = 0
        val cands = ScanLogic.locationCandidates(target, dets)
        scanExecutor.execute { runScan(target, cands, w, h) }
    }

    /** The label crops of one target from every photo of a burst, sharpest first. */
    private class Shot(val crop: Bitmap, val sharpness: Double)

    /**
     * Finds each expected label (normalised, already zoomed) in every photo of the burst and cuts it
     * out at full resolution. Returns, per label, its crops sorted sharpest first.
     */
    private fun cropBurst(stills: List<Still>, expected: List<RectF>, cls: Int, margin: Float, minWidth: Int, frameAspect: Float): List<List<Shot>> {
        val out = expected.map { mutableListOf<Shot>() }
        for (still in stills) {
            val (small, scale) = still.preview()
            val dets = detect(small).map {
                Detection(it.cls, it.score, RectF(it.box.left / scale, it.box.top / scale, it.box.right / scale, it.box.bottom / scale))
            }
            val taken = mutableSetOf<Detection>()
            val aspect = still.width.toFloat() / still.height
            expected.forEachIndexed { i, e ->
                val at = ScanLogic.relocate(ScanLogic.toStillAspect(e, frameAspect, aspect), dets, cls, still.width, still.height, taken)
                val crop = still.crop(at, margin, minWidth) ?: return@forEachIndexed
                out[i] += Shot(crop, ScanLogic.sharpness(crop))
            }
        }
        return out.map { l -> l.sortedByDescending { it.sharpness } }
    }

    /**
     * Reads a label from its burst crops: text from the sharpest crop, barcodes from the first crop
     * that decodes (a slightly softer photo sometimes decodes when the sharpest does not).
     */
    private fun readBurst(shots: List<Shot>): Pair<Bitmap, LabelRead>? {
        if (shots.isEmpty()) return null
        val first = LabelReader.read(shots[0].crop)
        if (first.codes.isNotEmpty()) return shots[0].crop to first
        for (s in shots.drop(1)) {
            val r = LabelReader.read(s.crop, wantText = false)
            if (r.codes.isNotEmpty()) return s.crop to LabelRead(r.codes, first.text, r.engine)
        }
        return shots[0].crop to first
    }

    /**
     * Switches to the lens that gives [zoom] with the most real detail, lets the user aim at a label of
     * class [cls] and returns a burst of photos. Used when the label needs more zoom than it can get
     * while staying in the picture (a phone zooms around the middle of the frame).
     */
    private fun aimAndBurst(cls: Int, zoom: Float, diag: JSONObject, key: String): List<Still> {
        val lens = planLens(zoom)
        val used = if (switchLens(lens)) lens else { switchLens(lenses.firstOrNull()); lenses.firstOrNull() }
        val magnification = used?.magnification ?: 1f
        val cam = camera
        val maxRatio = cam?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1f
        val ratio = (zoom / magnification).coerceIn(1f, maxRatio)
        runCatching { cam?.cameraControl?.setZoomRatio(ratio)?.get(2, TimeUnit.SECONDS) }
        diag.put("${key}_lens", "%.1fx".format(magnification))
        diag.put("${key}_digital_zoom", ratio.toDouble())

        val trigger = CompletableFuture<Boolean>()
        aimClass = cls; aimStable = 0; aimTrigger = trigger
        ui { overlay.clear(); overlay.reticle = true; btnScan.text = "Take photo now"; autoScan.visibility = View.INVISIBLE }
        state = State.AIMING
        val go = try {
            trigger.get(60, TimeUnit.SECONDS)
        } catch (e: java.util.concurrent.TimeoutException) {
            false
        } finally {
            state = State.CAPTURING
            aimTrigger = null
            ui { overlay.reticle = false; overlay.clear(); btnScan.text = "Scan now"; autoScan.visibility = View.VISIBLE }
        }
        if (!go) throw CancelledScan()

        // Focus on the label that was aimed at (or the middle of the frame).
        val t = ScanLogic.centreMost(lastDets.filter { it.cls == cls }, lastW, lastH)
        focusAt(t?.let { it.box.centerX() / lastW } ?: 0.5f, t?.let { it.box.centerY() / lastH } ?: 0.5f)
        ui { status.text = "Taking $BURST photos. Hold steady…" }
        return burst()
    }

    private class CancelledScan : RuntimeException()

    /**
     * Crops the labels of class [cls] from an aimed burst: the labels found in the first photo
     * (nearest the middle first, at most four) are followed through the other photos. When the
     * detector finds none, the whole middle of the photo is kept so the readers can still try.
     */
    private fun cropAimed(stills: List<Still>, cls: Int, margin: Float, minWidth: Int): List<List<Shot>> {
        val first = stills.first()
        val (small, _) = first.preview()
        val found = detect(small).filter { it.cls == cls }
            .sortedBy { Math.abs(it.box.centerX() - small.width / 2f) / small.width + Math.abs(it.box.centerY() - small.height / 2f) / small.height }
            .take(ScanLogic.MAX_CANDIDATES)
            .map { ScanLogic.normalise(it.box, small.width, small.height) }
        val expected = found.ifEmpty { listOf(RectF(0.12f, 0.2f, 0.88f, 0.8f)) }
        val aspect = first.width.toFloat() / first.height
        return cropBurst(stills, expected, cls, if (found.isEmpty()) 0f else margin, minWidth, aspect).filter { it.isNotEmpty() }
    }

    private fun runScan(target: Detection, cands: List<LocationCandidate>, w: Int, h: Int) {
        try {
            val frameAspect = w.toFloat() / h
            val diag = JSONObject()
            diag.put("lenses", JSONArray(lenses.map { "%.1fx".format(it.magnification) }))
            val found = mutableListOf<LocationRead>()
            var zLoc = 1f
            var locPhotos = 0
            if (cands.isNotEmpty()) {
                val boxes = cands.map { ScanLogic.normalise(it.box, w, h) }
                val mine = cands.indices.filter { cands[it].belongs }.map { boxes[it] }.ifEmpty { boxes }
                val group = ScanLogic.union(mine)
                zLoc = (LOCATION_LABEL_FILL / mine.map { it.width() }.average().toFloat().coerceAtLeast(0.01f)).coerceIn(1f, MAX_ZOOM)
                val reachable = ScanLogic.zoomFor(group, MAX_ZOOM)
                if ((planLens(zLoc)?.magnification ?: 1f) < 1.8f && reachable >= 0.75f * zLoc) {
                    // Close enough: zoom automatically, the labels stay in the picture.
                    switchLens(lenses.firstOrNull())
                    ui { overlay.clear(); status.text = "Zooming in on the location label…" }
                    zLoc = zoomTo(group)
                    ui { status.text = "Taking $BURST photos of the location label. Hold steady…" }
                    val stills = burst()
                    locPhotos = stills.size
                    ui { status.text = "Reading the location label…" }
                    cropBurst(stills, boxes.map { ScanLogic.zoomRect(it, zLoc) }, ScanLogic.LOCATION_LABEL, 0.25f, 700, frameAspect)
                        .forEachIndexed { i, shots ->
                            val (img, read) = readBurst(shots) ?: return@forEachIndexed
                            found += LocationRead(img, read, cands[i].belongs)
                        }
                    diag.put("location_lens", "1.0x"); diag.put("location_digital_zoom", zLoc.toDouble())
                } else {
                    val stills = aimAndBurst(ScanLogic.LOCATION_LABEL, zLoc, diag, "location")
                    locPhotos = stills.size
                    ui { status.text = "Reading the location label…" }
                    cropAimed(stills, ScanLogic.LOCATION_LABEL, 0.25f, 700).forEachIndexed { i, shots ->
                        val (img, read) = readBurst(shots) ?: return@forEachIndexed
                        found += LocationRead(img, read, i == 0)   // the label that was aimed at comes first
                    }
                }
                diag.put("location_photos", locPhotos)
            }

            val labelBox = ScanLogic.normalise(target.box, w, h)
            var zPal = (PALLET_LABEL_FILL / labelBox.width().coerceAtLeast(0.01f)).coerceIn(1f, MAX_ZOOM)
            val reachable = ScanLogic.zoomFor(labelBox, MAX_ZOOM)
            val stills: List<Still>
            val shots: List<Shot>
            if ((planLens(zPal)?.magnification ?: 1f) < 1.8f && reachable >= 0.75f * zPal) {
                switchLens(lenses.firstOrNull())
                ui { overlay.clear(); status.text = "Zooming in on the pallet label…" }
                zPal = zoomTo(labelBox)
                ui { status.text = "Taking $BURST photos of the pallet label. Hold steady…" }
                stills = burst()
                ui { status.text = "Reading the pallet label…" }
                shots = cropBurst(stills, listOf(ScanLogic.zoomRect(labelBox, zPal)), ScanLogic.PALLET_LABEL, 0.06f, 0, frameAspect)[0]
                diag.put("pallet_lens", "1.0x"); diag.put("pallet_digital_zoom", zPal.toDouble())
            } else {
                stills = aimAndBurst(ScanLogic.PALLET_LABEL, zPal, diag, "pallet")
                ui { status.text = "Reading the pallet label…" }
                shots = cropAimed(stills, ScanLogic.PALLET_LABEL, 0.06f, 0).firstOrNull() ?: emptyList()
            }
            val result = readBurst(shots)
            restoreMainLens()

            val s0 = stills.firstOrNull()
            diag.put("photo_width", s0?.width ?: 0)
            diag.put("photo_height", s0?.height ?: 0)
            diag.put("photo_source", s0?.source ?: "none")
            diag.put("pallet_photos", stills.size)
            diag.put("pallet_sharpness", JSONArray(shots.map { it.sharpness.toInt() }))
            diag.put("pallet_label_px", result?.first?.width ?: 0)
            val summary = "Lenses ${lenses.joinToString(" ") { "%.1fx".format(it.magnification) }.ifEmpty { "main only" }} · " +
                "location ${"%.1f".format(zLoc)}x on ${diag.optString("location_lens", "-")} lens · pallet ${"%.1f".format(zPal)}x on ${diag.optString("pallet_lens", "-")} lens · " +
                "photo ${s0?.width ?: 0}×${s0?.height ?: 0} (${s0?.source ?: "none"}) · pallet label ${result?.first?.width ?: 0} px wide · " +
                "sharpest of ${shots.size} (scores ${shots.joinToString(", ") { it.sharpness.toInt().toString() }})"
            ui { showResult(found, result?.first, result?.second, zLoc, zPal, diag, summary) }
        } catch (e: CancelledScan) {
            restoreMainLens()
            ui { backToSearch(1500) }
        } catch (e: Exception) {
            restoreMainLens()
            ui {
                Toast.makeText(this, "Scan failed: ${e.message ?: e.javaClass.simpleName}", Toast.LENGTH_LONG).show()
                backToSearch(1500)
            }
        }
    }

    private fun restoreMainLens() {
        resetZoom()
        switchLens(lenses.firstOrNull())
        resetZoom()
    }

    /** Focuses and meters on a point of the live picture (0..1) and waits for the lens to settle. */
    private fun focusAt(x: Float, y: Float) {
        val cam = camera ?: return
        val display = previewView.display ?: return
        val point = DisplayOrientedMeteringPointFactory(display, cam.cameraInfo, 1f, 1f)
            .createPoint(x.coerceIn(0.05f, 0.95f), y.coerceIn(0.05f, 0.95f))
        val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
            .setAutoCancelDuration(6, TimeUnit.SECONDS).build()
        runCatching { cam.cameraControl.startFocusAndMetering(action).get(2500, TimeUnit.MILLISECONDS) }
        Thread.sleep(200)
    }

    /**
     * Zooms as far as [target] (normalised, at zoom 1) stays in the picture, focuses and meters on it,
     * and waits until the lens has settled. Returns the zoom that was set.
     */
    private fun zoomTo(target: RectF): Float {
        val cam = camera ?: return 1f
        val maxZoom = (cam.cameraInfo.zoomState.value?.maxZoomRatio ?: 1f).coerceAtMost(8f)
        val zoom = ScanLogic.zoomFor(target, maxZoom)
        runCatching { cam.cameraControl.setZoomRatio(zoom).get(2, TimeUnit.SECONDS) }
        val display = previewView.display
        if (display != null) {
            val z = ScanLogic.zoomRect(target, zoom)
            val point = DisplayOrientedMeteringPointFactory(display, cam.cameraInfo, 1f, 1f)
                .createPoint(z.centerX().coerceIn(0.05f, 0.95f), z.centerY().coerceIn(0.05f, 0.95f))
            val action = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF or FocusMeteringAction.FLAG_AE)
                .setAutoCancelDuration(6, TimeUnit.SECONDS).build()
            runCatching { cam.cameraControl.startFocusAndMetering(action).get(2500, TimeUnit.MILLISECONDS) }
        }
        Thread.sleep(250)
        return zoom
    }

    private fun resetZoom() {
        runCatching { camera?.cameraControl?.setZoomRatio(1f) }
        runCatching { camera?.cameraControl?.cancelFocusAndMetering() }
    }

    /** Several photos in a row at the current zoom. Hands shake and focus hunts, so one of them is usually sharper. */
    private fun burst(): List<Still> {
        val stills = mutableListOf<Still>()
        val started = SystemClock.elapsedRealtime()
        for (i in 0 until BURST) {
            // High-resolution photos are slow on some devices: stop after 6 s but keep at least two.
            if (i >= 2 && SystemClock.elapsedRealtime() - started > 6000) break
            runCatching { grabStill() }.getOrNull()?.let { stills += it }
        }
        if (stills.isEmpty()) throw IllegalStateException("the camera did not return a photo")
        return stills
    }

    /** A full-resolution photo at the current zoom; falls back to the next preview frame. */
    private fun grabStill(): Still {
        val capture = imageCapture
        if (capture != null) {
            val future = CompletableFuture<Still>()
            capture.takePicture(captureExecutor, object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    try {
                        val rotation = image.imageInfo.rotationDegrees
                        if (image.format == ImageFormat.JPEG) {
                            val buf = image.planes[0].buffer
                            val bytes = ByteArray(buf.remaining())
                            buf.get(bytes)
                            future.complete(Still.fromJpeg(bytes, image.cropRect, rotation))
                        } else {
                            future.complete(Still.fromBitmap(LiveActivity.rotate(image.toBitmap(), rotation), "photo"))
                        }
                    } catch (e: Throwable) {
                        future.completeExceptionally(e)
                    } finally {
                        image.close()
                    }
                }
                override fun onError(exception: ImageCaptureException) {
                    future.completeExceptionally(exception)
                }
            })
            try {
                return future.get(8, TimeUnit.SECONDS)
            } catch (e: Exception) {
                // fall through to a preview frame
            }
        }
        val future = CompletableFuture<Bitmap>()
        frameRequest = future
        return Still.fromBitmap(future.get(4, TimeUnit.SECONDS), "preview frame, low resolution")
    }

    // ---------------------------------------------------------------- 3. confirm and save

    private fun showResult(found: List<LocationRead>, crop: Bitmap?, read: LabelRead?, zLoc: Float, zPal: Float, diag: JSONObject, summary: String) {
        state = State.RESULT
        locations = found; palletCrop = crop; palletRead = read; zoomLocation = zLoc; zoomPallet = zPal
        // Preselect the best-ranked label that was actually read.
        selected = found.indexOfFirst { it.belongs && it.read.bestValue().isNotBlank() }
            .let { if (it >= 0) it else found.indexOfFirst { l -> l.read.bestValue().isNotBlank() } }
            .let { if (it >= 0) it else if (found.isNotEmpty()) 0 else -1 }
        locationId.setText(if (selected >= 0) found[selected].read.bestValue() else "")
        val matching = found.count { it.belongs }
        locationHint.text = when {
            found.isEmpty() -> "No location label was found for this pallet. Type the location or retake."
            found.size == 1 -> "One location label found. Check it is the right one."
            matching == 1 -> "The first label matches this pallet. Tap another one if it is wrong."
            else -> "${found.size} location labels near this pallet. Tap the one that belongs to it."
        }
        drawCandidates()
        palletImage.setImageBitmap(crop)
        palletCodes.text = when {
            crop == null -> "The pallet label could not be captured. Retake."
            read == null || read.codes.isEmpty() -> "No barcode read on the device. The image is saved for server-side decoding."
            else -> read.codes.joinToString("\n") { "${it.format}: ${it.value}" }
        }
        palletText.text = read?.text ?: ""
        diagnostics = diag
        diagText.text = summary
        resultPanel.scrollTo(0, 0)
        resultPanel.visibility = View.VISIBLE
        status.text = "Check the result"
    }

    private fun drawCandidates() {
        candidatesRow.removeAllViews()
        val d = resources.displayMetrics.density
        locations.forEachIndexed { i, loc ->
            val cell = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding((6 * d).toInt(), (6 * d).toInt(), (6 * d).toInt(), (6 * d).toInt())
                background = GradientDrawable().apply {
                    cornerRadius = 8 * d
                    setColor(Color.rgb(34, 34, 34))
                    setStroke((3 * d).toInt(), if (i == selected) Color.rgb(255, 220, 0) else Color.rgb(68, 68, 68))
                }
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT)
                    .apply { marginEnd = (8 * d).toInt() }
                setOnClickListener {
                    selected = i
                    locationId.setText(loc.read.bestValue())
                    drawCandidates()
                }
            }
            cell.addView(ImageView(this).apply {
                setImageBitmap(loc.image)
                adjustViewBounds = true
                contentDescription = "Location label ${i + 1}"
                layoutParams = LinearLayout.LayoutParams((200 * d).toInt(), LinearLayout.LayoutParams.WRAP_CONTENT)
                    .apply { height = (90 * d).toInt() }
                scaleType = ImageView.ScaleType.FIT_CENTER
            })
            cell.addView(TextView(this).apply {
                text = loc.read.bestValue().ifBlank { "not readable" }
                setTextColor(Color.WHITE); textSize = 16f
                maxWidth = (200 * d).toInt()
            })
            cell.addView(TextView(this).apply {
                text = if (loc.belongs) "matches this pallet" else "nearby"
                setTextColor(if (loc.belongs) Color.rgb(90, 220, 140) else Color.rgb(170, 170, 170)); textSize = 12f
            })
            candidatesRow.addView(cell)
        }
    }

    private fun saveResult() {
        val id = locationId.text.toString().trim()
        val crop = palletCrop
        val locs = locations
        val sel = selected
        val read = palletRead
        val diag = diagnostics
        scanExecutor.execute {
            val ok = runCatching { ScanStore.save(this, id, sel, locs, crop, read, zoomLocation, zoomPallet, diag) }.isSuccess
            ui {
                Toast.makeText(this, if (ok) "Saved${if (id.isNotEmpty()) " $id" else ""}" else "Could not save the scan", Toast.LENGTH_SHORT).show()
                if (ok) backToSearch(2500)
            }
        }
    }

    /** Back to the live view; automatic scanning pauses for [pauseMs] so the same pallet is not scanned twice. */
    private fun backToSearch(pauseMs: Long) {
        resultPanel.visibility = View.GONE
        palletImage.setImageBitmap(null)
        candidatesRow.removeAllViews()
        locations = emptyList(); palletCrop = null; palletRead = null; selected = -1
        stableFrames = 0
        cooldownUntil = SystemClock.elapsedRealtime() + pauseMs
        status.text = "Point at the next pallet"
        state = State.SEARCH
    }

    private fun ui(block: () -> Unit) = runOnUiThread { if (!isFinishing && !isDestroyed) block() }

    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        when (state) {
            State.RESULT -> backToSearch(1000)
            State.AIMING -> aimTrigger?.complete(false)
            else -> super.onBackPressed()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        scanExecutor.shutdown()
        captureExecutor.shutdown()
    }
}
