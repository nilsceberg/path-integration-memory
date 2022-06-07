import { useEffect, useMemo, useState } from "react";
import { Bar, Line, BarChart, CartesianGrid, Legend, Tooltip, XAxis, YAxis, ComposedChart, ResponsiveContainer, Scatter } from "recharts";

export default function Layers(props) {
    const { state } = props;
    if (!state) return null;

    const { tb1, cpu4 } = state.layers;

    return (
        <>
            <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={tb1.map((a, i)  => ({ i: i+1, TB1: a }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="i" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Scatter dataKey="TB1" fill="#8884d8" />
                </ComposedChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={250}>
                <ComposedChart data={cpu4.map((a, i)  => ({ i: i+1, CPU4: a }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="i" />
                    <YAxis domain={[0.3, 0.7]}/>
                    <Tooltip />
                    <Legend />
                    <Scatter dataKey="CPU4" fill="#8884d8" />
                </ComposedChart>
            </ResponsiveContainer>
        </>
    );
}