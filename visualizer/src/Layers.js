import { useEffect, useMemo, useState } from "react";
import { Bar, Line, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis, ComposedChart, ResponsiveContainer } from "recharts";

export default function Layers(props) {
    const { state } = props;

    const [update, setUpdate] = useState(0);

    useEffect(() => {
        const timeout = setTimeout(() => setUpdate(update + 1), 250);
        return () => clearTimeout(timeout);
    }, [update])

    const tb1 = useMemo(
        () => state?.tb1 || [],
        [update]
    );

    const cpu4 = useMemo(
        () => state?.cpu4 || [],
        [update]
    );

    //const tb1 = state?.tb1 || [];
    //const cpu4 = state?.cpu4 || [];

    return (
        <>
            <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={tb1.map((a, i)  => ({ i: i+1, TB1: a }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="i" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line dataKey="TB1" fill="#8884d8" />
                </ComposedChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={cpu4.map((a, i)  => ({ i: i+1, CPU4: a }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="i" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line dataKey="CPU4" fill="#8884d8" />
                </ComposedChart>
            </ResponsiveContainer>
        </>
    );
}