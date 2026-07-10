import skullVideo from "../../assets/skull.mp4";
import "./NexCaseCommandDashboard.css";

/**
 * Skull video interlude slide.
 */
export function NexCaseCommandDashboard() {
  return (
    <section className="nx-command" aria-label="Forensic skull video">
      <video
        className="nx-command-video"
        src={skullVideo}
        autoPlay
        muted
        loop
        playsInline
        preload="auto"
      />
    </section>
  );
}

export default NexCaseCommandDashboard;
