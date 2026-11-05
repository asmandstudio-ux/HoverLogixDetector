package nl.hoverlogix.detect

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
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
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Live camera feed with on-device detection. "Freeze & correct" opens the frame in the editor. */
class LiveActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private lateinit var btnModel: Button
    private lateinit var btnFreeze: Button

    private lateinit var analysisExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var camera: Camera? = null

    @Volatile private var confidence = 0.35f
    @Volatile private var kind = Detectors.Kind.FAST
    @Volatile private var lastFrame: Bitmap? = null
    private var frames = 0
    private var fpsWindowStart = SystemClock.elapsedRealtime()
    private var fps = 0f

    private val requestCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startCamera() else {
            Toast.makeText(this, "Camera permission is needed for the live feed", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Labels.load(this)
        setContentView(R.layout.activity_live)
        previewView = findViewById(R.id.previewView)
        overlay = findViewById(R.id.overlay)
        status = findViewById(R.id.status)
        btnModel = findViewById(R.id.btnModel)
        btnFreeze = findViewById(R.id.btnFreeze)

        previewView.scaleType = PreviewView.ScaleType.FILL_CENTER
        overlay.scaleMode = OverlayView.ScaleMode.FILL
        analysisExecutor = Executors.newSingleThreadExecutor()

        confidence = Settings.confidence(this)
        kind = Settings.liveModel(this)
        btnModel.text = kind.label
        btnModel.setOnClickListener {
            kind = if (kind == Detectors.Kind.FAST) Detectors.Kind.ACCURATE else Detectors.Kind.FAST
            Settings.setLiveModel(this, kind)
            btnModel.text = kind.label
        }

        val seek = findViewById<SeekBar>(R.id.confSeek)
        val confLabel = findViewById<TextView>(R.id.confLabel)
        seek.progress = (confidence * 100).toInt()
        confLabel.text = "Min. confidence ${seek.progress}%"
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                val v = p.coerceAtLeast(5)
                confidence = v / 100f
                confLabel.text = "Min. confidence $v%"
            }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) = Settings.setConfidence(this@LiveActivity, confidence)
        })

        btnFreeze.setOnClickListener { freezeAndCorrect() }

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

            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                        .build()
                )
                .build()

            provider.unbindAll()
            try {
                camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
                imageCapture = capture
            } catch (e: Exception) {
                // Some devices cannot run three streams at once; fall back to preview + analysis.
                provider.unbindAll()
                camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
                imageCapture = null
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(image: ImageProxy) {
        try {
            val bmp = rotate(image.toBitmap(), image.imageInfo.rotationDegrees)
            val detector = Detectors.get(this, kind)
            val dets = detector.detect(bmp, confidence)
            lastFrame = bmp
            frames++
            val now = SystemClock.elapsedRealtime()
            if (now - fpsWindowStart >= 1000) {
                fps = frames * 1000f / (now - fpsWindowStart)
                frames = 0; fpsWindowStart = now
            }
            val text = summary(dets) + String.format("   %.1f fps · %d ms", fps, detector.lastInferenceMs)
            runOnUiThread {
                overlay.setDetections(dets, bmp.width, bmp.height)
                status.text = text
            }
        } catch (e: Exception) {
            runOnUiThread { status.text = "Detection error: ${e.message}" }
        } finally {
            image.close()
        }
    }

    private fun freezeAndCorrect() {
        btnFreeze.isEnabled = false
        val capture = imageCapture
        if (capture == null) {
            openEditor(lastFrame)
            return
        }
        // Take a full-resolution still for review: small labels need the pixels.
        capture.takePicture(analysisExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bmp = try { rotate(image.toBitmap(), image.imageInfo.rotationDegrees) } catch (e: Exception) { lastFrame }
                image.close()
                runOnUiThread { openEditor(bmp) }
            }
            override fun onError(exception: ImageCaptureException) {
                runOnUiThread { openEditor(lastFrame) }
            }
        })
    }

    private fun openEditor(frame: Bitmap?) {
        btnFreeze.isEnabled = true
        if (frame == null) {
            Toast.makeText(this, "No frame yet", Toast.LENGTH_SHORT).show(); return
        }
        val path = FrameStore.put(this, frame)
        startActivity(Intent(this, EditorActivity::class.java).apply {
            putExtra(EditorActivity.EXTRA_FRAME_PATH, path)
            putExtra(EditorActivity.EXTRA_SOURCE, "live")
        })
    }

    override fun onDestroy() {
        super.onDestroy()
        analysisExecutor.shutdown()
    }

    companion object {
        fun rotate(src: Bitmap, degrees: Int): Bitmap {
            if (degrees % 360 == 0) return src
            val m = Matrix().apply { postRotate(degrees.toFloat()) }
            return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
        }

        fun summary(dets: List<Detection>): String {
            if (dets.isEmpty()) return "Nothing detected"
            return dets.groupingBy { it.cls }.eachCount().toSortedMap()
                .entries.joinToString("  ") { "${Labels.name(it.key)} ${it.value}" }
        }
    }
}
