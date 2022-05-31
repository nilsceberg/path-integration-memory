import { Paper } from "@mui/material";

export default function World(props) {
    const { state } = props;
    if (!state) return null;

    const [x, y] = state.position;

    return (
        <Paper elevation={0} sx={{ width: "100%", height: "100%" }}>
            <div style={{
                position: "absolute",
                left: "50%",
                top: "50%",
                width: "16px",
                height: "16px",
                marginLeft: -8 + x * 5,
                marginTop: -8 - y * 5,
                borderRadius: "8px",
                background: "white",
            }}/>
        </Paper>
    )
}