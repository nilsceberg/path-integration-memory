import useWebSocket from "react-use-websocket";
import { useCallback, useEffect, useMemo, useState } from "react";
import useAnimationFrame from "use-animation-frame";
import { Card, CardContent, CardHeader, CssBaseline} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import GridLayout, { WidthProvider } from "react-grid-layout";
import { useDebounce, useInterval } from "usehooks-ts";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";

import World from "./World";
import Layers from "./Layers";
import Plot from "./Plot";

const Grid = WidthProvider(GridLayout);

function Window(props) {
    const { title, children, sx } = props;
    return (
        <Card sx={{ width: "100%", height: "100%", display: "flex", flexDirection: "column", ...sx }}>
            <CardHeader className="windowResizeHandle" sx={{
                    background: theme =>
                        theme.palette.mode === "light"
                        ? theme.palette.grey[200]
                        : theme.palette.grey[900],
                    padding: "4px",
                }} subheader={title}/>
            <CardContent sx={{ flexGrow: "1" }}>{ children }</CardContent>
        </Card>
    );
}

function Controls(props) {
    const { sendJsonMessage, readyState, state, dataRate, fps, tps } = props;
    if (readyState === 1) {
        return (
            <div id="debugInfo">
                <div style={{ width: 200 }}>
                    data rate: {(dataRate / 1000 * 8).toFixed(0)}&nbsp;kb/s
                </div>
                <div style={{ width: 200 }}>
                    framerate: {fps}&nbsp;Hz
                </div>
                <div style={{ width: 300 }}>
                    server framerate: {tps}&nbsp;Hz
                </div>
            </div>
        );
    }
    else {
        return (
            <>
                disconnected
            </>
        );
    }
}

function App() {
    const [bytesReceived, setBytesReceived] = useState(0);
    const [dataRate, setDataRate] = useState(0);

    const [frames, setFrames] = useState(0);
    const [ticks, setTicks] = useState(0);

    const [fps, setFps] = useState(0);
    const [tps, setTps] = useState(0);

    const [lastMessage, setLastMessage] = useState(null);
    const [realtimeState, setRealtimeState] = useState(null);
    const [state, setState] = useState(null);

    const onMessage = useCallback(event => {
        const data = JSON.parse(event.data);
        setLastMessage(data); // Is it JSON?
        setBytesReceived(bytesReceived => bytesReceived + (event.data.length || 0));
        setTicks(ticks => ticks + 1);
    }, [bytesReceived, ticks]);

    const updatePeriodic = useCallback(() => {
        setBytesReceived(0);
        setFrames(0);
        setTicks(0);
        setDataRate(bytesReceived);
        setFps(frames);
        setTps(ticks);
    }, [bytesReceived, frames, ticks, realtimeState]);

    const updateDebouncedState = useCallback(() => {
        setState(realtimeState);
    }, [realtimeState]);

    const {
        sendJsonMessage,
        readyState,
    } = useWebSocket("ws://localhost:8001", {
        retryOnError: true,
        reconnectInterval: 1000,
        reconnectAttempts: 10000000,
        shouldReconnect: () => true,
        onMessage: onMessage,
    });

    useInterval(updatePeriodic, 1000);
    useInterval(updateDebouncedState, 500);

    useAnimationFrame(() => {
        setRealtimeState(lastMessage);
        setFrames(frames => frames + 1);
    }, [lastMessage, frames]);

    const theme = createTheme({
        palette: {
            mode: "dark"
        }
    });

    const layout = [
        { i: "controls", x: 0, y: 0, w: 12, h: 1 },
        { i: "world", x: 0, y: 1, w: 6, h: 6 },
        { i: "tb1", x: 6, y: 0, w: 6, h: 3 },
        { i: "cpu4", x: 6, y: 3, w: 6, h: 3 },
        { i: "tn1", x: 0, y: 6, w: 6, h: 3 },
        { i: "tn2", x: 6, y: 6, w: 6, h: 3 },
    ];

    const windows = useMemo(() => ({
        controls: <Window title="Controls">
            <Controls state={realtimeState} readyState={readyState} sendJsonMessage={sendJsonMessage} dataRate={dataRate} tps={tps} fps={fps}/>
        </Window>,
        world: <Window title="World">
            <World state={realtimeState}/>
        </Window>,
        tb1: <Window title="TB1 / Delta7">
            <Plot name="tb1" data={[ {y: state?.layers.TB1, type: "line"} ]} layout={{ yaxis: { range: [0, 1] } }}/>
        </Window>,
        cpu4: <Window title="CPU4 / PFN">
            <Plot name="cpu4" data={[ {y: state?.layers.CPU4, type: "line"} ]} layout={{ yaxis: { range: [0, 1] } }}/>
        </Window>,
        tn1: <Window title="TN1">
            <Plot name="tn1" data={[ {y: state?.layers.TN1, type: "bar"} ]} layout={{ yaxis: { range: [0, 1] } }}/>
        </Window>,
        tn2: <Window title="TN2">
            <Plot name="tn2" data={[ {y: state?.layers.TN2, type: "bar"} ]} layout={{ yaxis: { range: [0, 1] } }}/>
        </Window>,
    }), [realtimeState, state, readyState, sendJsonMessage]);

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline/>
            <Grid layout={layout} cols={12} rowHeight={90} draggableHandle=".windowResizeHandle" isResizable>
                {Object.entries(windows).map(([key, window]) => <div key={key}>{window}</div>)}
            </Grid>
        </ThemeProvider>
    );
}

export default App;
