import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
  ResponsiveContainer,
} from 'recharts';

export default function GaugeChart({ percentage }) {
  const pct = Math.min(Math.max(Number(percentage) || 0, 0), 100);

  // High-vibrancy light theme colors
  let color = '#e11d48'; // Rose/Red for needs work
  if (pct >= 80) color = '#059669'; // Emerald for strong match
  else if (pct >= 60) color = '#d97706'; // Amber for moderate match

  const data = [{ value: pct, fill: color }];

  return (
    <div className="gauge-wrapper">
      <ResponsiveContainer width="100%" height={175}>
        <RadialBarChart
          cx="50%"
          cy="75%"
          innerRadius="72%"
          outerRadius="102%"
          barSize={14}
          data={data}
          startAngle={180}
          endAngle={0}
        >
          <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
          <RadialBar
            background={{ fill: 'rgba(226, 232, 240, 0.6)' }}
            dataKey="value"
            angleAxisId={0}
            cornerRadius={8}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      <div className="gauge-score" style={{ color }}>{pct}%</div>
      <div className="gauge-label">Overall Match Score</div>
    </div>
  );
}
