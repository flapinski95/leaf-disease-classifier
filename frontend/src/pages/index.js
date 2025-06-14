import { useState } from 'react';
import styles from '@/styles/Home.module.css';

export default function Home() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

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
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Wysyłanie...' : 'Wyślij'}
        </button>
      </form>

      {result && (
        <div className={styles.result}>
          {result.error ? (
            <p className={styles.error}>{result.error}</p>
          ) : (
            <>
              <p><strong>Predykcja:</strong> {result.label}</p>
              <p><strong>Pewność:</strong> {(result.confidence * 100).toFixed(2)}%</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}