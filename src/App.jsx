import { useMemo, useState } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  GitBranch,
  Play,
  RotateCcw,
  Sigma,
  Sparkles,
  Target,
} from 'lucide-react';

const POINTS = [
  { id: 'A', x: 18, y: 72, label: 1 },
  { id: 'B', x: 28, y: 83, label: 1 },
  { id: 'C', x: 39, y: 66, label: 1 },
  { id: 'D', x: 52, y: 76, label: 1 },
  { id: 'E', x: 62, y: 58, label: 1 },
  { id: 'F', x: 76, y: 70, label: 1 },
  { id: 'G', x: 82, y: 48, label: 1 },
  { id: 'H', x: 16, y: 34, label: -1 },
  { id: 'I', x: 30, y: 46, label: -1 },
  { id: 'J', x: 44, y: 28, label: -1 },
  { id: 'K', x: 55, y: 49, label: -1 },
  { id: 'L', x: 68, y: 37, label: -1 },
  { id: 'M', x: 72, y: 62, label: -1 },
  { id: 'N', x: 88, y: 31, label: -1 },
];

const ROUND_COUNT = 3;
const EPSILON = 1e-5;
const CLASS_LABELS = {
  1: '+1 signal',
  '-1': '-1 signal',
};

function predictWithStump(point, stump) {
  const value = point[stump.axis];
  const positiveSide = value >= stump.threshold ? 1 : -1;
  return stump.polarity * positiveSide;
}

function buildThresholds(axis) {
  const values = [...new Set(POINTS.map((point) => point[axis]))].sort((a, b) => a - b);
  const thresholds = [0];

  for (let index = 0; index < values.length - 1; index += 1) {
    thresholds.push((values[index] + values[index + 1]) / 2);
  }

  thresholds.push(100);
  return thresholds;
}

function findBestStump(weights) {
  let best = null;

  for (const axis of ['x', 'y']) {
    for (const threshold of buildThresholds(axis)) {
      for (const polarity of [1, -1]) {
        const stump = { axis, threshold, polarity };
        const predictions = POINTS.map((point) => predictWithStump(point, stump));
        const error = predictions.reduce((total, prediction, index) => {
          return total + (prediction === POINTS[index].label ? 0 : weights[index]);
        }, 0);

        if (!best || error < best.error) {
          best = { ...stump, predictions, error };
        }
      }
    }
  }

  return best;
}

function computeBoostingRounds() {
  let weights = POINTS.map(() => 1 / POINTS.length);
  const rounds = [];

  for (let roundIndex = 0; roundIndex < ROUND_COUNT; roundIndex += 1) {
    const learner = findBestStump(weights);
    const boundedError = Math.min(0.49999, Math.max(EPSILON, learner.error));
    const alpha = 0.5 * Math.log((1 - boundedError) / boundedError);
    const nextWeights = weights.map((weight, index) => {
      const point = POINTS[index];
      const prediction = learner.predictions[index];
      return weight * Math.exp(-alpha * point.label * prediction);
    });
    const normalizer = nextWeights.reduce((total, weight) => total + weight, 0);

    rounds.push({
      ...learner,
      alpha,
      weightsBefore: weights,
      weightsAfter: nextWeights.map((weight) => weight / normalizer),
      misses: learner.predictions.map((prediction, index) => prediction !== POINTS[index].label),
    });

    weights = nextWeights.map((weight) => weight / normalizer);
  }

  return rounds;
}

function ensembleScore(point, rounds) {
  return rounds.reduce((score, round) => {
    return score + round.alpha * predictWithStump(point, round);
  }, 0);
}

function accuracyForPredictions(predictions) {
  const correct = predictions.filter((prediction, index) => prediction === POINTS[index].label).length;
  return correct / POINTS.length;
}

function accuracyForEnsemble(rounds) {
  const correct = POINTS.filter((point) => Math.sign(ensembleScore(point, rounds)) === point.label).length;
  return correct / POINTS.length;
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatNumber(value, digits = 2) {
  return value.toFixed(digits);
}

function describeStump(round) {
  const axisLabel = round.axis === 'x' ? 'x-position' : 'y-position';
  const leftPrediction = round.polarity === 1 ? -1 : 1;
  const rightPrediction = round.polarity === 1 ? 1 : -1;
  return `${axisLabel} >= ${formatNumber(round.threshold, 1)} predicts ${CLASS_LABELS[rightPrediction]}; otherwise ${CLASS_LABELS[leftPrediction]}`;
}

function App() {
  const rounds = useMemo(() => computeBoostingRounds(), []);
  const [selectedRound, setSelectedRound] = useState(0);
  const activeRound = rounds[selectedRound];
  const visibleRounds = rounds.slice(0, selectedRound + 1);
  const weakAccuracy = accuracyForPredictions(activeRound.predictions);
  const ensembleAccuracy = accuracyForEnsemble(visibleRounds);

  const goPrevious = () => setSelectedRound((round) => Math.max(0, round - 1));
  const goNext = () => setSelectedRound((round) => Math.min(rounds.length - 1, round + 1));
  const reset = () => setSelectedRound(0);

  return (
    <main className="app-shell">
      <header className="site-header" aria-label="Site header">
        <a className="brand" href="#top" aria-label="Boosting Lab home">
          <span className="brand-mark" aria-hidden="true">
            <GitBranch size={18} strokeWidth={2.2} />
          </span>
          Boosting Lab
        </a>
        <nav className="site-nav" aria-label="Page navigation">
          <a href="#demo">Demo</a>
          <a href="#steps">Steps</a>
          <a href="#compare">Compare</a>
        </nav>
      </header>

      <section id="demo" className="demo-layout" aria-labelledby="demo-title">
        <TeachingPanel
          title="Boosting turns weak guesses into a strong ensemble"
          round={activeRound}
          selectedRound={selectedRound}
        />

        <section className="visual-panel" aria-label="Interactive boosting visualization">
          <ControlBar
            selectedRound={selectedRound}
            roundCount={rounds.length}
            onPrevious={goPrevious}
            onNext={goNext}
            onReset={reset}
            onSelectRound={setSelectedRound}
          />
          <BoostingCanvas
            points={POINTS}
            rounds={rounds}
            selectedRound={selectedRound}
            visibleRounds={visibleRounds}
          />
          <div className="plot-footer">
            <div>
              <span className="legend-dot positive" aria-hidden="true" />
              +1 signal
            </div>
            <div>
              <span className="legend-dot negative" aria-hidden="true" />
              -1 signal
            </div>
            <div>
              <span className="legend-ring" aria-hidden="true" />
              mistake this round
            </div>
          </div>
        </section>

        <SummaryPanel
          round={activeRound}
          rounds={rounds}
          visibleRounds={visibleRounds}
          selectedRound={selectedRound}
          weakAccuracy={weakAccuracy}
          ensembleAccuracy={ensembleAccuracy}
        />
      </section>

      <section id="steps" className="explain-band" aria-labelledby="steps-title">
        <div className="section-heading">
          <h2 id="steps-title">Step through the rounds</h2>
          <p>
            Boosting repeats the same small lesson: train where the current model struggles,
            then add the new learner to the vote with a confidence score.
          </p>
        </div>

        <div className="step-grid">
          <article className="step-card">
            <span className="step-icon" aria-hidden="true">
              <Target size={22} />
            </span>
            <h3>Fit a weak learner</h3>
            <p>
              Each round picks the best decision stump for the current point weights, even if
              that stump is only a little better than guessing.
            </p>
          </article>
          <article className="step-card">
            <span className="step-icon amber" aria-hidden="true">
              <Sparkles size={22} />
            </span>
            <h3>Lift the misses</h3>
            <p>
              Misclassified points get heavier, so the next stump is pulled toward the cases
              the ensemble has not explained yet.
            </p>
          </article>
          <article className="step-card">
            <span className="step-icon graphite" aria-hidden="true">
              <Sigma size={22} />
            </span>
            <h3>Vote by confidence</h3>
            <p>
              Low-error stumps earn larger alpha weights. The final prediction is the sign of
              the weighted vote across all selected learners.
            </p>
          </article>
        </div>
      </section>

      <section id="compare" className="comparison-section" aria-labelledby="compare-title">
        <div className="comparison-copy">
          <h2 id="compare-title">Weak learner versus boosted ensemble</h2>
          <p>
            A single stump draws one simple boundary. Boosting stacks several of those simple
            boundaries, paying extra attention to points that were previously hard to classify.
          </p>
        </div>
        <ComparisonBars
          weakAccuracy={weakAccuracy}
          ensembleAccuracy={ensembleAccuracy}
          selectedRound={selectedRound}
        />
      </section>
    </main>
  );
}

function TeachingPanel({ title, round, selectedRound }) {
  return (
    <aside className="teaching-panel">
      <h1 id="demo-title">{title}</h1>
      <p className="lede">
        Boosting builds an ensemble one weak learner at a time. The learner for each round is
        trained on weighted data, so earlier mistakes become louder in the next round.
      </p>

      <div className="formula-block">
        <h2>Round {selectedRound + 1}</h2>
        <p>{describeStump(round)}</p>
        <div className="formula">
          <span>alpha</span>
          <strong>0.5 ln((1 - error) / error)</strong>
        </div>
        <div className="formula">
          <span>next weight</span>
          <strong>w x exp(-alpha y h(x))</strong>
        </div>
      </div>

      <div className="teaching-note">
        <strong>Read the plot:</strong> larger circles have more influence. A black ring means
        the selected weak learner got that point wrong, so its next-round weight rises.
      </div>
    </aside>
  );
}

function ControlBar({ selectedRound, roundCount, onPrevious, onNext, onReset, onSelectRound }) {
  return (
    <div className="control-bar">
      <div className="round-tabs" role="tablist" aria-label="Boosting rounds">
        {Array.from({ length: roundCount }, (_, index) => (
          <button
            key={index}
            type="button"
            className={selectedRound === index ? 'round-tab active' : 'round-tab'}
            onClick={() => onSelectRound(index)}
            role="tab"
            aria-selected={selectedRound === index}
          >
            Round {index + 1}
          </button>
        ))}
      </div>

      <div className="control-actions" aria-label="Round controls">
        <button type="button" className="icon-button" onClick={onPrevious} disabled={selectedRound === 0}>
          <ChevronLeft size={18} />
          <span className="sr-only">Previous round</span>
        </button>
        <button
          type="button"
          className="primary-action"
          onClick={onNext}
          disabled={selectedRound === roundCount - 1}
        >
          <Play size={16} />
          Next
        </button>
        <button type="button" className="icon-button" onClick={onNext} disabled={selectedRound === roundCount - 1}>
          <ChevronRight size={18} />
          <span className="sr-only">Next round</span>
        </button>
        <button type="button" className="icon-button" onClick={onReset}>
          <RotateCcw size={17} />
          <span className="sr-only">Reset rounds</span>
        </button>
      </div>
    </div>
  );
}

function BoostingCanvas({ points, rounds, selectedRound, visibleRounds }) {
  const width = 720;
  const height = 500;
  const padding = 54;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const activeRound = rounds[selectedRound];
  const maxWeight = Math.max(...activeRound.weightsBefore);
  const minWeight = Math.min(...activeRound.weightsBefore);
  const regions = buildRegions(visibleRounds);

  const sx = (value) => padding + (value / 100) * plotWidth;
  const sy = (value) => height - padding - (value / 100) * plotHeight;
  const scaleRadius = (weight) => {
    const normalized = (weight - minWeight) / Math.max(0.001, maxWeight - minWeight);
    return 8 + normalized * 13;
  };

  return (
    <div className="canvas-shell">
      <svg className="boosting-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="plot-title plot-desc">
        <title id="plot-title">Weighted training points and weak learner boundaries</title>
        <desc id="plot-desc">
          A scatter plot showing class labels, point weights, misclassified examples, and the
          selected weak learner for the current boosting round.
        </desc>

        <rect className="plot-bg" x={padding} y={padding} width={plotWidth} height={plotHeight} rx="8" />

        {Array.from({ length: 6 }, (_, index) => {
          const value = index * 20;
          return (
            <g key={value} className="grid-line">
              <line x1={sx(value)} x2={sx(value)} y1={padding} y2={height - padding} />
              <line x1={padding} x2={width - padding} y1={sy(value)} y2={sy(value)} />
            </g>
          );
        })}

        {regions.map((region) => (
          <rect
            key={`${region.x}-${region.y}`}
            className={region.prediction === 1 ? 'region positive' : 'region negative'}
            x={sx(region.x)}
            y={sy(region.y + region.size)}
            width={(region.size / 100) * plotWidth + 0.4}
            height={(region.size / 100) * plotHeight + 0.4}
          />
        ))}

        {visibleRounds.map((round, index) => (
          <StumpLine
            key={`${round.axis}-${round.threshold}-${index}`}
            round={round}
            index={index}
            active={index === selectedRound}
            sx={sx}
            sy={sy}
            padding={padding}
            height={height}
            width={width}
          />
        ))}

        {points.map((point, index) => {
          const missed = activeRound.misses[index];
          const weight = activeRound.weightsBefore[index];
          return (
            <g key={point.id} className="point-group">
              <circle
                className={point.label === 1 ? 'point positive' : 'point negative'}
                cx={sx(point.x)}
                cy={sy(point.y)}
                r={scaleRadius(weight)}
              />
              <circle
                className={missed ? 'point-ring missed' : 'point-ring'}
                cx={sx(point.x)}
                cy={sy(point.y)}
                r={scaleRadius(weight) + 3}
              />
              <text x={sx(point.x)} y={sy(point.y) + 4} textAnchor="middle">
                {point.id}
              </text>
            </g>
          );
        })}

        <text className="axis-label" x={width / 2} y={height - 14} textAnchor="middle">
          feature x
        </text>
        <text className="axis-label vertical" x={18} y={height / 2} textAnchor="middle">
          feature y
        </text>
      </svg>
    </div>
  );
}

function buildRegions(rounds) {
  const cellSize = 5;
  const cells = [];

  for (let x = 0; x < 100; x += cellSize) {
    for (let y = 0; y < 100; y += cellSize) {
      const score = ensembleScore({ x: x + cellSize / 2, y: y + cellSize / 2 }, rounds);
      cells.push({
        x,
        y,
        size: cellSize,
        prediction: score >= 0 ? 1 : -1,
      });
    }
  }

  return cells;
}

function StumpLine({ round, index, active, sx, sy, padding, height, width }) {
  const lineClass = active ? 'stump-line active' : 'stump-line';
  const labelX = round.axis === 'x' ? sx(round.threshold) + 10 : width - padding - 10;
  const labelY = round.axis === 'x' ? padding + 22 + index * 18 : sy(round.threshold) - 10;

  if (round.axis === 'x') {
    return (
      <g>
        <line className={lineClass} x1={sx(round.threshold)} x2={sx(round.threshold)} y1={padding} y2={height - padding} />
        <text className="stump-label" x={labelX} y={labelY}>
          {`h${index + 1}: x >= ${formatNumber(round.threshold, 1)}`}
        </text>
      </g>
    );
  }

  return (
    <g>
      <line className={lineClass} x1={padding} x2={width - padding} y1={sy(round.threshold)} y2={sy(round.threshold)} />
      <text className="stump-label" x={labelX} y={labelY} textAnchor="end">
        {`h${index + 1}: y >= ${formatNumber(round.threshold, 1)}`}
      </text>
    </g>
  );
}

function SummaryPanel({ round, rounds, visibleRounds, selectedRound, weakAccuracy, ensembleAccuracy }) {
  const maxAlpha = Math.max(...rounds.map((item) => item.alpha));
  const ensembleVotes = POINTS.reduce(
    (totals, point) => {
      const prediction = Math.sign(ensembleScore(point, visibleRounds));
      totals[prediction === 1 ? 'positive' : 'negative'] += 1;
      return totals;
    },
    { positive: 0, negative: 0 },
  );

  return (
    <aside className="summary-panel">
      <section className="metric-block">
        <h2>Weighted errors</h2>
        <div className="metric-row">
          <span>Error</span>
          <strong>{formatPercent(round.error)}</strong>
        </div>
        <div className="metric-row">
          <span>Alpha</span>
          <strong>{formatNumber(round.alpha)}</strong>
        </div>
        <div className="metric-track" aria-hidden="true">
          <span style={{ width: `${Math.min(100, round.error * 100)}%` }} />
        </div>
      </section>

      <section className="learner-list">
        <h2>Learner weights</h2>
        {rounds.map((item, index) => (
          <div
            key={`${item.axis}-${item.threshold}-${index}`}
            className={index <= selectedRound ? 'learner-row selected' : 'learner-row'}
            aria-label={`Round ${index + 1} alpha ${formatNumber(item.alpha)}`}
          >
            <span>Round {index + 1}</span>
            <span className="alpha-bar">
              <span style={{ width: `${(item.alpha / maxAlpha) * 100}%` }} />
            </span>
            <strong>{formatNumber(item.alpha)}</strong>
          </div>
        ))}
      </section>

      <section className="vote-block">
        <h2>Ensemble vote</h2>
        <div className="vote-scale">
          <span style={{ width: `${(ensembleVotes.positive / POINTS.length) * 100}%` }} />
        </div>
        <div className="vote-labels">
          <span>+1: {ensembleVotes.positive}</span>
          <span>-1: {ensembleVotes.negative}</span>
        </div>
        <div className="accuracy-pair">
          <div>
            <span>Weak learner</span>
            <strong>{formatPercent(weakAccuracy)}</strong>
          </div>
          <div>
            <span>Boosted ensemble</span>
            <strong>{formatPercent(ensembleAccuracy)}</strong>
          </div>
        </div>
      </section>
    </aside>
  );
}

function ComparisonBars({ weakAccuracy, ensembleAccuracy, selectedRound }) {
  return (
    <div className="comparison-bars" aria-label={`Round ${selectedRound + 1} comparison`}>
      <div className="comparison-row">
        <div>
          <span>Current weak learner</span>
          <strong>{formatPercent(weakAccuracy)}</strong>
        </div>
        <div className="wide-track">
          <span style={{ width: `${weakAccuracy * 100}%` }} />
        </div>
      </div>
      <div className="comparison-row emphasized">
        <div>
          <span>Boosted through round {selectedRound + 1}</span>
          <strong>{formatPercent(ensembleAccuracy)}</strong>
        </div>
        <div className="wide-track">
          <span style={{ width: `${ensembleAccuracy * 100}%` }} />
        </div>
      </div>
    </div>
  );
}

export default App;
