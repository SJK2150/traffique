import { useRef, useState, useEffect } from 'react'
import { Play, Pause, SkipForward, SkipBack, Maximize } from 'lucide-react'

const VideoPlayer = ({ 
  videoSrc, 
  onFrameChange,
  overlayRenderer 
}) => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [duration, setDuration] = useState(0)
  const [fps, setFps] = useState(30)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
      // Estimate FPS (default to 30 if can't determine)
      setFps(30)
    }

    const handleTimeUpdate = () => {
      const frame = Math.floor(video.currentTime * fps)
      setCurrentFrame(frame)
      onFrameChange?.(frame)
      
      // Draw overlay
      if (canvasRef.current && overlayRenderer) {
        const ctx = canvasRef.current.getContext('2d')
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
        overlayRenderer(ctx, frame)
      }
    }

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('timeupdate', handleTimeUpdate)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('timeupdate', handleTimeUpdate)
    }
  }, [fps, onFrameChange, overlayRenderer])

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return

    if (isPlaying) {
      video.pause()
    } else {
      video.play()
    }
    setIsPlaying(!isPlaying)
  }

  const skipFrames = (frames) => {
    const video = videoRef.current
    if (!video) return

    const newTime = video.currentTime + (frames / fps)
    video.currentTime = Math.max(0, Math.min(newTime, duration))
  }

  const toggleFullscreen = () => {
    const container = videoRef.current?.parentElement
    if (!container) return

    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      container.requestFullscreen()
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-4">
      {/* Video Container */}
      <div className="relative bg-forest-900 rounded-xl overflow-hidden">
        <video
          ref={videoRef}
          src={videoSrc}
          className="w-full"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          width={1280}
          height={720}
        />
      </div>

      {/* Controls */}
      <div className="card p-4">
        <div className="flex items-center gap-4">
          <button onClick={() => skipFrames(-10)} className="btn-ghost p-2">
            <SkipBack className="w-5 h-5" />
          </button>
          
          <button onClick={togglePlay} className="btn btn-primary">
            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
          </button>
          
          <button onClick={() => skipFrames(10)} className="btn-ghost p-2">
            <SkipForward className="w-5 h-5" />
          </button>

          <div className="flex-1">
            <input
              type="range"
              min="0"
              max={duration}
              value={videoRef.current?.currentTime || 0}
              onChange={(e) => {
                if (videoRef.current) {
                  videoRef.current.currentTime = e.target.value
                }
              }}
              className="w-full"
            />
          </div>

          <span className="text-sm text-forest-700 font-mono">
            {formatTime(videoRef.current?.currentTime || 0)} / {formatTime(duration)}
          </span>

          <span className="text-sm text-forest-700 font-mono">
            Frame: {currentFrame}
          </span>

          <button onClick={toggleFullscreen} className="btn-ghost p-2">
            <Maximize className="w-5 h-5" />
          </button>
        </div>
      </div>
    </div>
  )
}

export default VideoPlayer
