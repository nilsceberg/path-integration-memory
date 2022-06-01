import { Paper } from "@mui/material";
import Spritesheet from 'react-responsive-spritesheet';
import beeImg from './bee_11.png';

export default function World(props) {
    const { state } = props;
    if (!state) return null;

    const [x, y] = state.position;
    const heading = state.decoded_heading * 180 / Math.PI + 90;

    return (
        <Paper elevation={0} sx={{ width: "100%", height: "100%", backgroundColor: "#323232" }}>
            <Spritesheet
                image={beeImg}
                widthFrame={23}
                heightFrame={23}
                steps={12}
                fps={24}
                autoplay={true}
                loop={true}
                style={{
                    position: "absolute",
                    left: "50%",
                    top: "50%",
                    width: "40px",
                    height: "40px",
                    marginLeft: -8 + x * 5,
                    marginTop: -8 - y * 5,
                    transform: `rotate(${heading.toFixed(0)}deg)`
                }}/>
        </Paper>
    )
}