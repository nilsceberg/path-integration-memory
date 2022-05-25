import useWebSocket from "react-use-websocket";
import { useEffect } from "react";
import { Card, CardContent, CardHeader, CssBaseline} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import GridLayout, { WidthProvider } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";

const Grid = WidthProvider(GridLayout);

function Window(props) {
    const { title, children, sx } = props;
    return (
        <Card sx={{ weight: "100%", height: "100%", ...sx }}>
            <CardHeader className="windowResizeHandle" sx={{
                    background: theme =>
                        theme.palette.mode === "light"
                        ? theme.palette.grey[200]
                        : theme.palette.grey[800],
                }} title={title}/>
            <CardContent>{ children }</CardContent>
        </Card>
    );
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

    const bee = state
        ? <div style={{ width: 8, height: 8, background: "black", position: 'fixed', left: 200 + 10 * state.position[0], top: 200 - 10 * state.position[1] }}/>
        : null;

    const layout = [
        { i: "world", x: 0, y: 0, w: 6, h: 6 },
        { i: "layers", x: 6, y: 0, w: 6, h: 6 },
    ];

    const theme = createTheme({
        palette: {
            mode: "dark"
        }
    });

    const windows = {
        world: <Window title="World">
            content
        </Window>,
        layers: <Window title="Layers">
            content
        </Window>,
    };

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline/>
            <Grid layout={layout} cols={12} draggableHandle=".windowResizeHandle" isResizable>
                {Object.entries(windows).map(([key, window]) => <div key={key}>{window}</div>)}
            </Grid>
        </ThemeProvider>
    );
}

export default App;
