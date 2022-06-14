import { Paper } from "@mui/material";
import Spritesheet from 'react-responsive-spritesheet';
import beeImg from './bee_11.png';

export default function World(props) {
    const { state } = props;
    if (!state) return null;

    const scale = 20.0;

    const [x, y] = state.position;
    const heading = -state.heading * 180 / Math.PI + 90;

    const [ex, ey] = state.estimated_position;
    const eh = -state.estimated_heading[0] * 180 / Math.PI + 90;

    const width = 40;
    const height = 40;

    return (
        <Paper elevation={0} sx={{ width: "100%", height: "100%", backgroundColor: "#323232" }}>
            <div style={{color: "gray", position: "absolute", left: "50%", top: "50%", marginLeft: "-20px"}}>HOME</div>
            <Spritesheet
                image={beeImg}
                widthFrame={23}
                heightFrame={23}
                steps={12}
                fps={24}
                autoplay={true}
                loop={true}
                style={{
                    opacity: 0.5,
                    position: "absolute",
                    left: "50%",
                    top: "50%",
                    width: `${width}px`,
                    height: `${height}px`,
                    marginLeft: -width/2 + ex * scale, // * 25,
                    marginTop: -height/2 - ey * scale, // * 25,
                    transform: `rotate(${eh.toFixed(0)}deg)`,
                    transformOrigin: '50% 50%',
                }}/>
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
                    width: `${width}px`,
                    height: `${height}px`,
                    marginLeft: -width/2 + x * scale,
                    marginTop: -height/2 - y * scale,
                    transform: `rotate(${heading.toFixed(0)}deg)`,
                    transformOrigin: '50% 50%',
                }}/>
        </Paper>
    )
}