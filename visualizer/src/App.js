import useWebSocket from "react-use-websocket";
import { useEffect } from "react";
import { Card, CardContent, CardHeader, CssBaseline} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import GridLayout, { WidthProvider } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";

import World from "./World";
import Layers from "./Layers";

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
    const { sendJsonMessage, readyState, state } = props;
    if (readyState === 1) {
        return (
            <>
                connected
            </>
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

    const layout = [
        { i: "controls", x: 0, y: 0, w: 12, h: 1 },
        { i: "world", x: 0, y: 1, w: 6, h: 5 },
        { i: "layers", x: 6, y: 1, w: 6, h: 5 },
    ];

    const theme = createTheme({
        palette: {
            mode: "dark"
        }
    });

    const windows = {
        controls: <Window title="Controls">
            <Controls state={state} readyState={readyState} sendJsonMessage={sendJsonMessage}/>
        </Window>,
        world: <Window title="World">
            <World state={state}/>
        </Window>,
        layers: <Window title="Layers">
            <Layers state={state}/>
        </Window>,
    };

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline/>
            <Grid layout={layout} cols={12} rowHeight={120} draggableHandle=".windowResizeHandle" isResizable>
                {Object.entries(windows).map(([key, window]) => <div key={key}>{window}</div>)}
            </Grid>
        </ThemeProvider>
    );
}

export default App;
