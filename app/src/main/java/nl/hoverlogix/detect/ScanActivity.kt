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
import androidx.camera.core.Camera
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

    private enum class State { SEARCH, CAPTURING, RESULT }

    companion object {
        /** Photos per label; the sharpest one that decodes is used. */
        const val BURST = 4
        /** Smallest pallet-label width in photo pixels at which barcodes are normally readable. */
        const val MIN_LABEL_PX = 300
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
        btnScan.setOnClickListener { startScan(null) }
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
            val provider = providerFuture.get()
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
            // real pixels on the label; digital zoom adds none). Then the normal maximum, then no photos.
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
                    provider.unbindAll()
                    camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
                    imageCapture = capture
                    break
                } catch (e: Exception) {
                    // try the next option
                }
            }
            if (camera == null) {
                provider.unbindAll()
                camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // ---------------------------------------------------------------- 1. search

    private fun analyze(image: ImageProxy) {
        try {
            val request = frameRequest
            if (state != State.SEARCH && request == null) return
            val bmp = LiveActivity.rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            if (request != null) {           // a still was asked for and ImageCapture is not available
                frameRequest = null
                request.complete(bmp)
                return
            }
            val dets = detect(bmp)
            lastDets = dets; lastW = bmp.width; lastH = bmp.height
            val labels = dets.filter { it.cls == ScanLogic.PALLET_LABEL }
            val hasLocation = dets.any { it.cls == ScanLogic.LOCATION_LABEL }
            val target = ScanLogic.centreMost(labels, bmp.width, bmp.height)
            val cands = if (target != null) ScanLogic.locationCandidates(target, dets) else emptyList()

            // Real pixels the pallet label will have in the photo (digital zoom adds none).
            val labelPx = if (target != null) (target.box.width() / bmp.width * photoWidth(bmp.width < bmp.height)).toInt() else 0
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

    private fun runScan(target: Detection, cands: List<LocationCandidate>, w: Int, h: Int) {
        try {
            val frameAspect = w.toFloat() / h
            val diag = JSONObject()
            val found = mutableListOf<LocationRead>()
            var zLoc = 1f
            if (cands.isNotEmpty()) {
                ui { overlay.clear(); status.text = "Zooming in on the location label…" }
                val boxes = cands.map { ScanLogic.normalise(it.box, w, h) }
                // Zoom on the labels that belong to this pallet; the others are read from the same photos.
                val focusOn = cands.indices.filter { cands[it].belongs }.map { boxes[it] }.ifEmpty { boxes }
                zLoc = zoomTo(ScanLogic.union(focusOn))
                ui { status.text = "Taking $BURST photos of the location label. Hold steady…" }
                val stills = burst()
                ui { status.text = "Reading the location label…" }
                val crops = cropBurst(stills, boxes.map { ScanLogic.zoomRect(it, zLoc) }, ScanLogic.LOCATION_LABEL, 0.25f, 700, frameAspect)
                crops.forEachIndexed { i, shots ->
                    val (img, read) = readBurst(shots) ?: return@forEachIndexed
                    found += LocationRead(img, read, cands[i].belongs)
                }
                diag.put("location_photos", stills.size)
                diag.put("location_sharpness", JSONArray(crops.firstOrNull()?.map { it.sharpness.toInt() } ?: emptyList<Int>()))
            }

            ui { overlay.clear(); status.text = "Zooming in on the pallet label…" }
            val labelBox = ScanLogic.normalise(target.box, w, h)
            val zPal = zoomTo(labelBox)
            ui { status.text = "Taking $BURST photos of the pallet label. Hold steady…" }
            val stills = burst()
            ui { status.text = "Reading the pallet label…" }
            val shots = cropBurst(stills, listOf(ScanLogic.zoomRect(labelBox, zPal)), ScanLogic.PALLET_LABEL, 0.06f, 0, frameAspect)[0]
            val result = readBurst(shots)
            resetZoom()

            val s0 = stills.firstOrNull()
            diag.put("photo_width", s0?.width ?: 0)
            diag.put("photo_height", s0?.height ?: 0)
            diag.put("photo_source", s0?.source ?: "none")
            diag.put("pallet_photos", stills.size)
            diag.put("pallet_sharpness", JSONArray(shots.map { it.sharpness.toInt() }))
            diag.put("pallet_label_px", result?.first?.width ?: 0)
            diag.put("max_zoom", (camera?.cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1f).toDouble())
            val summary = "Photo ${s0?.width ?: 0}×${s0?.height ?: 0} (${s0?.source ?: "none"}) · zoom ${"%.1f".format(zLoc)}x / ${"%.1f".format(zPal)}x · " +
                "pallet label ${result?.first?.width ?: 0} px wide · sharpest of ${shots.size} (scores ${shots.joinToString(", ") { it.sharpness.toInt().toString() }})"
            ui { showResult(found, result?.first, result?.second, zLoc, zPal, diag, summary) }
        } catch (e: Exception) {
            resetZoom()
            ui {
                Toast.makeText(this, "Scan failed: ${e.message ?: e.javaClass.simpleName}", Toast.LENGTH_LONG).show()
                backToSearch(1500)
            }
        }
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
        if (state == State.RESULT) backToSearch(1000) else super.onBackPressed()
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
        scanExecutor.shutdown()
        captureExecutor.shutdown()
    }
}
