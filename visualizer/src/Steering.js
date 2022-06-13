import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

export default function Steering(props) {
    const { state } = props;
    const { tb1 } = state;

    const [figure, setFigure] = useState({});

    return (
        <>
            <Plot
                data={[
                    {
                        y: tb1,
                        type: "bar",
                    }
                ]}
                config={{
                }}
            />
        </>
    );
}
