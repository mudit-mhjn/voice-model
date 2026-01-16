(() => {
    const audio = document.getElementById("patientPlayback");
    if (!audio) return;
  
    let loading = false;
  
    const loadIfNeeded = async () => {
      if (audio.dataset.loaded === "true" || loading) return;
      loading = true;
      try {
        const resp = await fetch(`/dashboard/patient/${audio.dataset.patientId}/audio`);
        if (!resp.ok) throw new Error("Failed to load audio URL");
        const data = await resp.json();
        if (!data.url) throw new Error("Missing audio URL");
  
        audio.src = data.url;
        audio.dataset.loaded = "true";
        audio.load();
      } catch (err) {
        console.error(err);
        audio.dataset.loaded = "error";
      } finally {
        loading = false;
      }
    };
  
    const startPlayback = async () => {
      if (audio.dataset.loaded !== "true") {
        await loadIfNeeded();
      }
      if (audio.dataset.loaded === "true") {
        audio.play().catch(() => {});
      }
    };
  
    audio.addEventListener("click", startPlayback);
    audio.addEventListener("pointerdown", startPlayback);
    audio.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        startPlayback();
      }
    });
  })();