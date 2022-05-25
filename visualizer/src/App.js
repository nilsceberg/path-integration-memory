import './App.css';
import useWebSocket from "react-use-websocket";
import { useEffect } from 'react';

function App() {
  const {
    sendJsonMessage,
    lastJsonMessage,
    readyState
  } = useWebSocket("ws://localhost:8001", {
    retryOnError: true,
    reconnectInterval: 1000,
    reconnectAttempts: 10000000,
    shouldReconnect: () => true,
  });

  const state = lastJsonMessage;

  const bee = state
    ? <div style={{ width: 8, height: 8, background: "black", position: 'fixed', left: 200 + 10 * state.position[0], top: 200 - 10 * state.position[1] }}/>
    : null;

  return (
    <div className="App">
      { readyState }
      <br/>
      { bee }
    </div>
  );
}

export default App;
