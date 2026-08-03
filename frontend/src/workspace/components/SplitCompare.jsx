import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Side-by-side comparison with a draggable divider and synchronised zoom.
 *
 * The two images are ALWAYS the original and its enhancement, and the original
 * is always on the left. The layout never shows the enhanced image alone: the
 * whole point of the component is that neither image can be presented without
 * the other.
 *
 * Zoom and pan are shared. Comparing detail at different magnifications is how
 * an examiner accidentally reads an artifact as a feature, so the two panes are
 * a single viewport by construction.
 */
export function SplitCompare({ originalUrl, enhancedUrl, enhancedLabel }) {
  const [split, setSplit] = useState(0.5);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);
  const dragState = useRef(null);

  const clampZoom = (value) => Math.min(Math.max(value, 1), 12);

  const handleWheel = useCallback((event) => {
    event.preventDefault();
    setZoom((current) => clampZoom(current * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
  }, []);

  useEffect(() => {
    const node = containerRef.current;
    if (!node) return undefined;
    // React attaches wheel listeners passively; preventing scroll needs an
    // explicit non-passive listener.
    node.addEventListener("wheel", handleWheel, { passive: false });
    return () => node.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  useEffect(() => {
    // Reset the viewport when the images change: a pan position from the
    // previous image pair is meaningless on the new one.
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSplit(0.5);
  }, [originalUrl, enhancedUrl]);

  function beginPan(event) {
    if (event.target.dataset?.role === "divider") return;
    dragState.current = {
      kind: "pan",
      startX: event.clientX,
      startY: event.clientY,
      origin: { ...pan },
    };
  }

  function beginDivider(event) {
    event.stopPropagation();
    dragState.current = { kind: "divider" };
  }

  function handleMove(event) {
    const state = dragState.current;
    const node = containerRef.current;
    if (!state || !node) return;
    if (state.kind === "divider") {
      const rect = node.getBoundingClientRect();
      setSplit(Math.min(Math.max((event.clientX - rect.left) / rect.width, 0.05), 0.95));
    } else if (state.kind === "pan" && zoom > 1) {
      setPan({
        x: state.origin.x + (event.clientX - state.startX),
        y: state.origin.y + (event.clientY - state.startY),
      });
    }
  }

  function endDrag() {
    dragState.current = null;
  }

  const imageStyle = {
    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
    transformOrigin: "center center",
  };

  return (
    <div className="wk-split-wrap">
      <div
        ref={containerRef}
        className="wk-split"
        role="img"
        aria-label="Original and enhanced images, side by side with a movable divider"
        onPointerDown={beginPan}
        onPointerMove={handleMove}
        onPointerUp={endDrag}
        onPointerLeave={endDrag}
      >
        <div className="wk-split-pane" style={{ width: `${split * 100}%` }}>
          {originalUrl && <img src={originalUrl} alt="Original evidence" style={imageStyle} draggable={false} />}
          <span className="wk-split-tag original">ORIGINAL — evidence</span>
        </div>
        <div className="wk-split-pane right" style={{ width: `${(1 - split) * 100}%` }}>
          {enhancedUrl && <img src={enhancedUrl} alt="Enhanced result" style={imageStyle} draggable={false} />}
          <span className="wk-split-tag enhanced">{enhancedLabel || "ENHANCED"}</span>
        </div>
        <div
          className="wk-split-divider"
          data-role="divider"
          style={{ left: `${split * 100}%` }}
          onPointerDown={beginDivider}
          role="separator"
          aria-orientation="vertical"
          aria-label="Drag to reveal more of either image"
        />
      </div>

      <div className="wk-split-controls">
        <button type="button" className="wk-button ghost small" onClick={() => setZoom((z) => clampZoom(z / 1.5))}>
          −
        </button>
        <span className="wk-zoom-label">{zoom.toFixed(1)}×</span>
        <button type="button" className="wk-button ghost small" onClick={() => setZoom((z) => clampZoom(z * 1.5))}>
          +
        </button>
        <button
          type="button"
          className="wk-button ghost small"
          onClick={() => {
            setZoom(1);
            setPan({ x: 0, y: 0 });
            setSplit(0.5);
          }}
        >
          Reset view
        </button>
        <small>Scroll to zoom · drag to pan · both panes move together</small>
      </div>
    </div>
  );
}
