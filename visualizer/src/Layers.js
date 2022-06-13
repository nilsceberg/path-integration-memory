import { useEffect, useMemo, useState } from "react";
import Plotly from "plotly.js-dist-min";
import { useDebounce, useEffectOnce } from "usehooks-ts";

export default function Layers(props) {
    const { state } = props;

    const tb1 = state?.layers.TB1;

    const tb1Layout = useMemo(() => ({
        title: "TB1",
        yaxis: {
            range: [0.0, 1.0],
        },
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
            "layer-tb1",
            [
                //{
                //    y: tb1,
                //    type: "bar"
                //},
                {
                    y: tb1,
                    type: "line"
                },
            ],
            tb1Layout,
            {
                responsive: true,
            }
        );
    }, [tb1, tb1Layout]);

    return (
        <div style={{height: "100%"}} id="layer-tb1"/>
    );
}