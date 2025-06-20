import { useState } from 'react';
import styles from '@/styles/Home.module.css';

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    if (selectedFile) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    } else {
      setPreviewUrl(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);

    setLoading(true);
    setResult(null);

    try {
      const res = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      console.log("Odpowiedź z backendu:", data);
      setResult(data);
    } catch (err) {
      console.error(err);
      setResult({ error: 'Błąd serwera' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>Rozpoznawanie chorób roślin 🌿</h1>

      <form onSubmit={handleSubmit} className={styles.form}>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />
        <button type="submit" disabled={loading || !file}>
          {loading ? 'Wysyłanie...' : 'Wyślij'}
        </button>
      </form>

      {previewUrl && (
        <div className={styles.preview}>
          <h3>Podgląd obrazu:</h3>
          <img src={previewUrl} alt="Preview" className={styles.image} />
        </div>
      )}

      {loading && <p className={styles.loading}>Analiza obrazu, proszę czekać...</p>}

      {result && (
  <div className={styles.result}>
    {result.error ? (
      <p className={styles.error}>{result.error}</p>
    ) : (
      <>
        <h2>Top 3 predykcje:</h2>
        {Array.isArray(result.predictions) && result.predictions.length > 0 ? (
          <ul>
            {result.predictions
              .filter(pred => pred.confidence > 0.05)
              .map((pred, index) => (
                <li key={index}>
                  <strong>{index + 1}. {pred.label}</strong> – {(pred.confidence * 100).toFixed(2)}%
                </li>
              ))}
          </ul>
        ) : (
          <p className={styles.error}>Brak wiarygodnych predykcji.</p>
        )}

        {result.warning && (
          <p className={styles.warning}>{result.warning}</p>
        )}
      </>
    )}
  </div>
)}
    </div>
  );
}