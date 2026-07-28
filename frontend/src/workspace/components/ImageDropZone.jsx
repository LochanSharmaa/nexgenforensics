import { useEffect, useState } from "react";

const ACCEPTED = "image/jpeg,image/png,image/webp,image/bmp,image/tiff";

export function ImageDropZone({ label, hint, file, onChange, id }) {
  const [preview, setPreview] = useState("");

  useEffect(() => {
    if (!file) {
      setPreview("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    // Object URLs pin the file in memory until revoked; without this an
    // investigator working through a batch leaks every image they looked at.
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return (
    <label className="wk-drop" htmlFor={id}>
      <input
        id={id}
        type="file"
        accept={ACCEPTED}
        onChange={(event) => onChange(event.target.files?.[0] || null)}
        aria-label={label}
      />
      {preview ? (
        <img src={preview} alt={`Selected ${label.toLowerCase()}`} />
      ) : (
        <>
          <strong>{label}</strong>
          <small>{hint || "JPEG, PNG, WEBP, BMP, or TIFF"}</small>
        </>
      )}
    </label>
  );
}
