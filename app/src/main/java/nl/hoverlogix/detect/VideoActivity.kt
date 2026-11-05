package nl.hoverlogix.detect

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.media.MediaMetadataRetriever
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.Surface
import android.view.TextureView
import android.widget.Button
import android.widget.FrameLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/** Plays a recorded video with detections drawn on top; step frame by frame and correct any frame. */
class VideoActivity : AppCompatActivity(), TextureView.SurfaceTextureListener {

    private lateinit var uri: Uri
    private lateinit var container: FrameLayout
    private lateinit var textureView: TextureView
    private lateinit var overlay: OverlayView
    private lateinit var status: TextView
    private lateinit var timeText: TextView
    private lateinit var seek: SeekBar
    private lateinit var btnPlay: Button
    private lateinit var btnModel: Button

    private var player: MediaPlayer? = null
    private var surface: Surface? = null
    private var prepared = false
    private var userSeeking = false

    private var dispW = 16
    private var dispH = 9
    private var rotation = 0
    private var frameMs = 33
    private var videoName = "video"

    private lateinit var executor: ExecutorService
    private val handler = Handler(Looper.getMainLooper())
    @Volatile private var busy = false
    private var detectAt = 0L       // run one detection on the paused frame after this time
    @Volatile private var confidence = 0.35f
    @Volatile private var kind = Detectors.Kind.ACCURATE

    private val loop = object : Runnable {
        override fun run() {
            tick()
            handler.postDelayed(this, 40)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Labels.load(this)
        setContentView(R.layout.activity_video)
        uri = intent.data ?: run { finish(); return }

        container = findViewById(R.id.videoContainer)
        textureView = findViewById(R.id.textureView)
        overlay = findViewById(R.id.overlay)
        status = findViewById(R.id.status)
        timeText = findViewById(R.id.timeText)
        seek = findViewById(R.id.seek)
        btnPlay = findViewById(R.id.btnPlay)
        btnModel = findViewById(R.id.btnModel)
        overlay.scaleMode = OverlayView.ScaleMode.FIT
        executor = Executors.newSingleThreadExecutor()

        confidence = Settings.confidence(this)
        kind = Settings.videoModel(this)
        btnModel.text = kind.label
        btnModel.setOnClickListener {
            kind = if (kind == Detectors.Kind.FAST) Detectors.Kind.ACCURATE else Detectors.Kind.FAST
            Settings.setVideoModel(this, kind)
            btnModel.text = kind.label
            requestDetect()
        }

        readMetadata()
        container.post { layoutVideo() }
        container.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ -> layoutVideo() }
        textureView.surfaceTextureListener = this

        btnPlay.setOnClickListener { togglePlay() }
        findViewById<Button>(R.id.btnPrev).setOnClickListener { step(-1) }
        findViewById<Button>(R.id.btnNext).setOnClickListener { step(+1) }
        findViewById<Button>(R.id.btnCorrect).setOnClickListener { correctThisFrame() }

        val confSeek = findViewById<SeekBar>(R.id.confSeek)
        val confLabel = findViewById<TextView>(R.id.confLabel)
        confSeek.progress = (confidence * 100).toInt()
        confLabel.text = "Min. confidence ${confSeek.progress}%"
        confSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                val v = p.coerceAtLeast(5)
                confidence = v / 100f
                confLabel.text = "Min. confidence $v%"
                requestDetect()
            }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) = Settings.setConfidence(this@VideoActivity, confidence)
        })

        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, fromUser: Boolean) {
                if (fromUser) updateTime(p)
            }
            override fun onStartTrackingTouch(s: SeekBar) { userSeeking = true }
            override fun onStopTrackingTouch(s: SeekBar) {
                userSeeking = false
                player?.seekTo(s.progress.toLong(), MediaPlayer.SEEK_CLOSEST)
            }
        })
    }

    private fun readMetadata() {
        val mmr = MediaMetadataRetriever()
        try {
            mmr.setDataSource(this, uri)
            val w = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 16
            val h = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 9
            rotation = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            if (rotation % 180 != 0) { dispW = h; dispH = w } else { dispW = w; dispH = h }
            val fpsMeta = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE)?.toFloatOrNull()
            if (fpsMeta != null && fpsMeta in 1f..240f) frameMs = max(1, (1000f / fpsMeta).roundToInt())
        } catch (e: Exception) {
            Toast.makeText(this, "Could not read video: ${e.message}", Toast.LENGTH_LONG).show()
        } finally {
            mmr.release()
        }
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) videoName = c.getString(0)?.substringBeforeLast('.') ?: videoName
        }
        title = videoName
    }

    /** Size the video and the overlay to the same aspect-correct rectangle. */
    private fun layoutVideo() {
        val cw = container.width
        val ch = container.height
        if (cw == 0 || ch == 0) return
        val s = min(cw.toFloat() / dispW, ch.toFloat() / dispH)
        val w = (dispW * s).toInt()
        val h = (dispH * s).toInt()
        val lp = textureView.layoutParams as FrameLayout.LayoutParams
        if (lp.width == w && lp.height == h) return
        textureView.layoutParams = FrameLayout.LayoutParams(w, h, Gravity.CENTER)
        overlay.layoutParams = FrameLayout.LayoutParams(w, h, Gravity.CENTER)
    }

    // ---------- TextureView / MediaPlayer ----------

    override fun onSurfaceTextureAvailable(st: SurfaceTexture, width: Int, height: Int) {
        surface = Surface(st)
        val mp = MediaPlayer()
        player = mp
        try {
            mp.setDataSource(this, uri)
            mp.setSurface(surface)
            mp.setOnPreparedListener {
                prepared = true
                seek.max = it.duration
                it.seekTo(0L, MediaPlayer.SEEK_CLOSEST)  // show first frame
                updateTime(0)
            }
            mp.setOnSeekCompleteListener { requestDetect() }
            mp.setOnCompletionListener { btnPlay.text = "Play" }
            mp.setOnErrorListener { _, what, extra ->
                Toast.makeText(this, "Playback error $what/$extra", Toast.LENGTH_LONG).show(); true
            }
            mp.prepareAsync()
        } catch (e: Exception) {
            Toast.makeText(this, "Cannot play this video: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    override fun onSurfaceTextureSizeChanged(st: SurfaceTexture, width: Int, height: Int) {}
    override fun onSurfaceTextureDestroyed(st: SurfaceTexture): Boolean {
        releasePlayer(); return true
    }
    override fun onSurfaceTextureUpdated(st: SurfaceTexture) {}

    private fun releasePlayer() {
        player?.release(); player = null; prepared = false
        surface?.release(); surface = null
    }

    override fun onResume() {
        super.onResume()
        handler.post(loop)
    }

    override fun onPause() {
        super.onPause()
        handler.removeCallbacks(loop)
        player?.let { if (it.isPlaying) { it.pause(); btnPlay.text = "Play" } }
    }

    override fun onDestroy() {
        super.onDestroy()
        releasePlayer()
        executor.shutdown()
    }

    // ---------- controls ----------

    private fun togglePlay() {
        val mp = player ?: return
        if (!prepared) return
        if (mp.isPlaying) {
            mp.pause(); btnPlay.text = "Play"; requestDetect()
        } else {
            if (mp.currentPosition >= mp.duration - frameMs) mp.seekTo(0L, MediaPlayer.SEEK_CLOSEST)
            mp.start(); btnPlay.text = "Pause"
        }
    }

    private fun step(dir: Int) {
        val mp = player ?: return
        if (!prepared) return
        if (mp.isPlaying) { mp.pause(); btnPlay.text = "Play" }
        val target = (mp.currentPosition + dir * frameMs).coerceIn(0, mp.duration)
        mp.seekTo(target.toLong(), MediaPlayer.SEEK_CLOSEST)
        seek.progress = target
        updateTime(target)
    }

    private fun requestDetect() {
        detectAt = SystemClock.elapsedRealtime() + 120  // give the TextureView time to show the new frame
    }

    private fun tick() {
        val mp = player ?: return
        if (!prepared) return
        val playing = mp.isPlaying
        if (playing && !userSeeking) {
            seek.progress = mp.currentPosition
            updateTime(mp.currentPosition)
        }
        val pendingStill = detectAt != 0L && SystemClock.elapsedRealtime() >= detectAt
        if (busy || !textureView.isAvailable || !(playing || pendingStill)) return
        if (pendingStill) detectAt = 0L

        // Grab the displayed frame at a size close to the model input.
        val longSide = 1280f
        val s = min(1f, longSide / max(dispW, dispH))
        val bmp: Bitmap = textureView.getBitmap(max(1, (dispW * s).toInt()), max(1, (dispH * s).toInt())) ?: return
        busy = true
        val conf = confidence
        val k = kind
        executor.execute {
            val result = runCatching {
                val det = Detectors.get(this, k)
                det.detect(bmp, conf) to det.lastInferenceMs
            }
            runOnUiThread {
                result.onSuccess { (dets, ms) ->
                    overlay.setDetections(dets, bmp.width, bmp.height)
                    status.text = LiveActivity.summary(dets) + "   ${ms} ms"
                }.onFailure { status.text = "Detection error: ${it.message}" }
                busy = false
            }
        }
    }

    private fun updateTime(ms: Int) {
        val total = player?.duration ?: 0
        timeText.text = "${fmt(ms)} / ${fmt(total)}"
    }

    private fun fmt(ms: Int): String {
        val s = ms / 1000
        return String.format(Locale.US, "%d:%02d.%03d", s / 60, s % 60, ms % 1000)
    }

    // ---------- correction ----------

    private fun correctThisFrame() {
        val mp = player ?: return
        if (!prepared) return
        if (mp.isPlaying) { mp.pause(); btnPlay.text = "Play" }
        val posMs = mp.currentPosition
        val fallback = textureView.bitmap
        Toast.makeText(this, "Grabbing full-resolution frame…", Toast.LENGTH_SHORT).show()
        executor.execute {
            val frame = grabFullResFrame(posMs) ?: fallback
            runOnUiThread {
                if (frame == null) {
                    Toast.makeText(this, "Could not read this frame", Toast.LENGTH_SHORT).show()
                    return@runOnUiThread
                }
                val path = FrameStore.put(this, frame)
                startActivity(Intent(this, EditorActivity::class.java).apply {
                    putExtra(EditorActivity.EXTRA_FRAME_PATH, path)
                    putExtra(EditorActivity.EXTRA_SOURCE, "${videoName}_${posMs}ms")
                })
            }
        }
    }

    private fun grabFullResFrame(posMs: Int): Bitmap? {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(this, uri)
            var bmp = mmr.getFrameAtTime(posMs * 1000L, MediaMetadataRetriever.OPTION_CLOSEST) ?: return null
            // Some devices return the frame unrotated; match the on-screen orientation.
            val bmpLandscape = bmp.width > bmp.height
            val dispLandscape = dispW > dispH
            if (rotation % 180 != 0 && bmpLandscape != dispLandscape) bmp = LiveActivity.rotate(bmp, rotation)
            bmp
        } catch (e: Exception) {
            null
        } finally {
            mmr.release()
        }
    }
}
