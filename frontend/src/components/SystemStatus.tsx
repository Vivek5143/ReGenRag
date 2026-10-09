import { useState, useEffect } from "react";
import { fetchSystemStatus } from "../services/api";

interface SystemStatusProps {
  status?: {
    health: { status: string; service: string };
    ready: { status: string; service: string; database: string };
  };
}

export default function SystemStatus({ status }: SystemStatusProps) {
  const [systemStatus, setSystemStatus] = useState<SystemStatusProps["status"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    if (!status) {
      const fetchStatus = async () => {
        try {
          const result = await fetchSystemStatus();
          setSystemStatus(result);
          setLastChecked(new Date());
        } catch (err) {
          setError("Unable to connect to backend");
          setSystemStatus(undefined);
        }
      };

      fetchStatus();
      const interval = setInterval(fetchStatus, 10000);
      return () => clearInterval(interval);
    } else {
      setSystemStatus(status);
      setLastChecked(new Date());
    }
  }, [status]);

  if (!systemStatus && !status) {
    return null;
  }

  const currentStatus = systemStatus || status;
  if (!currentStatus) {
    return null;
  }

  const getStatusClass = (statusStr: string): string => {
    switch (statusStr.toLowerCase()) {
      case "ok":
      case "ready":
        return "ok";
      case "degraded":
        return "warning";
      case "error":
        return "error";
      default:
        return "ok";
    }
  };

  return (
    <div className="card" style={{ marginBottom: "1rem" }}>
      <div className="status-grid">
        <div className="status-item">
          <div className="status-label">API</div>
          <div className={`status-value ${getStatusClass(currentStatus.health.status)}`}>
            {currentStatus.health.status.toUpperCase()}
          </div>
          <div className="status-detail">{currentStatus.health.service}</div>
        </div>

        <div className="status-item">
          <div className="status-label">Database</div>
          <div className={`status-value ${getStatusClass(currentStatus.ready.status)}`}>
            {currentStatus.ready.status.toUpperCase()}
          </div>
          <div className="status-detail">{currentStatus.ready.database}</div>
        </div>
      </div>

      {lastChecked && (
        <p className="muted" style={{ textAlign: "center", fontSize: "0.75rem", marginTop: "0.75rem" }}>
          Last checked: {lastChecked.toLocaleTimeString()}
        </p>
      )}

      {error && (
        <p className="error" style={{ textAlign: "center", marginTop: "0.5rem" }}>
          {error}
        </p>
      )}
    </div>
  );
}
