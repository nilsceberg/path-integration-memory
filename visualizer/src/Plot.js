import { useEffect, useMemo } from "react";
import Plotly from "plotly.js-dist-min";

export default function Plot(props) {
    const { data, name, layout } = props;

    const id = `plot-${name}`;

    const layout_ = useMemo(() => ({
        ...layout,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: {
            b: 30,
            l: 30,
            t: 30,
            r: 20,
            pad: 10,
        }
    }), []);

    useEffect(() => {
        Plotly.react(
            id,
            data,
            layout_,
            {
                responsive: true,
            }
        );
    }, [data, layout_]);

    return (
        <div style={{height: "100%"}} id={id}/>
    );
}