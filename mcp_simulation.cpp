// Compile with g++ -std=c++20 -O3 mcp_simulation.cpp -o mcp_simulation

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <span>
#include <format>
#include <cstdint>

// ============================================================================
// 1. Model Parameters (Spain First Wave - Amaro & Orce, 2022)
// ============================================================================
struct ModelParams
{
    double Ro = 8.0;    // Average interaction range per day (Eq. 4)
    int L = 300;        // Lattice linear size (total cells N = L^2)
    double sigma = 0.1; // Interaction cross section times density (rho * sigma)
    double T = 20.0;    // Temperature / individual mobility (Eq. 6)
    double pi = 0.2;    // Probability of infection upon contact
    double tr = 20.7;   // Removal inflection time in days (Eq. 9)
    double b = 4.5;     // Removal width/rate parameter in days
    double pd = 0.3;    // Death probability among removed individuals
    int delta = 1;      // Simulation step in days
};

// Compartmental states for each individual in the grid
enum class CellState : uint8_t
{
    SUSCEPTIBLE = 0,
    INFECTED = 1,
    RECOVERED = 2,
    DEAD = 3
};

// ============================================================================
// 2. Removal Probability Function
//    Evaluated daily at night as a Bernoulli parameter
// ============================================================================
constexpr double removalFunction(double daysInfected, double tr, double b) noexcept
{
    return 1.0 / (1.0 + std::exp((tr - daysInfected) / b));
}

// ============================================================================
// 3. MCP Simulation Engine
// ============================================================================
class MCPSimulation
{
public:
    ModelParams params;
    int numberOfCells;

    std::vector<CellState> state;
    std::vector<int> infectionTime;

    // Fast 64-bit Mersenne Twister RNG
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> uniform01{0.0, 1.0};

    explicit MCPSimulation(ModelParams p, uint64_t seed = 42)
        : params(p), numberOfCells(p.L * p.L), rng(seed)
    {
        reset();
    }

    void reset()
    {
        using enum CellState;

        state.assign(numberOfCells, SUSCEPTIBLE);
        infectionTime.assign(numberOfCells, 0);

        // Center cell k0 is initially infected
        const int centerI = params.L / 2;
        const int centerJ = params.L / 2;
        const int k0 = centerI * params.L + centerJ;

        state[k0] = INFECTED;
        infectionTime[k0] = 0;
    }

    // Stochastic rounding: floor(x) + 1 with probability equal to the
    // fractional part of x. E[result] = x exactly, so no bias is introduced.
    [[nodiscard]] int stochasticRound(double x)
    {
        const double fl = std::floor(x);
        return static_cast<int>(fl) + (uniform01(rng) < (x - fl) ? 1 : 0);
    }

    // Picks a random target cell inside the square of side 2R centered at k.
    // A continuous point is sampled in the square and floored to a cell index, so
    // even small ranges (R < 0.5) can reach the neighbours. Cell (i, j) covers
    // [i, i+1) x [j, j+1), so the square is centered at (i + 0.5, j + 0.5).
    // Out-of-lattice targets are moved to the border.
    [[nodiscard]] int pickTarget(int k, double R)
    {
        const int L = params.L;
        const int i = k / L;
        const int j = k % L;

        int i2 = static_cast<int>(std::floor(i + 0.5 - R + 2.0 * R * uniform01(rng)));
        int j2 = static_cast<int>(std::floor(j + 0.5 - R + 2.0 * R * uniform01(rng)));

        i2 = std::clamp(i2, 0, L - 1);
        j2 = std::clamp(j2, 0, L - 1);
        return i2 * L + j2;
    }

    // Advances the epidemic by one time step (t)
    void timeStep(int t)
    {
        using enum CellState;

        // Daily distributions based on current parameters
        std::exponential_distribution<double> rDist(1.0 / params.Ro);

        // Normalized discrete Planck distribution (Geometric with p = 1 - exp(-1/T), support in N_0)
        const double geomP = 1.0 - std::exp(-1.0 / params.T);
        std::geometric_distribution<int> nDist(geomP);

        for (int k = 0; k < numberOfCells; ++k)
        {
            if (state[k] != INFECTED || infectionTime[k] == t)
            {
                continue;
            }

            // Step II: Sample interaction range R.
            const double R = rDist(rng);

            // Step III: Sample discrete energy/mobility n.
            const int n = nDist(rng);

            // N_int = n * rho * sigma with rho = 1.0, randomly rounded
            const int nInteraction = stochasticRound(n * 1.0 * params.sigma);

            // Step IV: Contagion process (dart-throwing, sampling with replacement)
            for (int shot = 0; shot < nInteraction; ++shot)
            {
		// We are supposing that we can interact with a target more than once in a given time step.
                const int x = pickTarget(k, R);

                if (state[x] == SUSCEPTIBLE)
                {
                    if (uniform01(rng) < params.pi)
                    {
                        state[x] = INFECTED;
                        infectionTime[x] = t; // Target cell x is infected
                    }
                }
            }

            // Step V: Daily removal evaluated at night
            const double remProb = removalFunction(t - infectionTime[k], params.tr, params.b);
            if (uniform01(rng) < remProb)
            {
                // Step VI: Outcome of removal (death or recovery)
                if (uniform01(rng) < params.pd)
                {
                    state[k] = DEAD;
                }
                else
                {
                    state[k] = RECOVERED;
                }
            }
        }
    }

    [[nodiscard]] std::span<const CellState> getGridState() const noexcept
    {
        return state;
    }
};

// ============================================================================
// 4. Main Execution Loop
// ============================================================================
int main()
{
    using enum CellState;

    ModelParams spainParams{
        .Ro = 8.0,
        .L = 300,
        .sigma = 0.1,
        .T = 20.0,
        .pi = 0.2,
        .tr = 20.7,
        .b = 4.5,
        .pd = 0.3,
        .delta = 1};

    MCPSimulation sim(spainParams, 42);

    constexpr int totalDays = 200;
    int prevDead = 0;

    std::cout << std::format("{:>5} | {:>12} | {:>10} | {:>10} | {:>8} | {:>12}\n",
                             "Day", "Susceptible", "Infected", "Recovered", "Dead", "Daily Deaths");
    std::cout << std::string(72, '-') << '\n';

    for (int day = 1; day <= totalDays; day += sim.params.delta)
    {
        // Enact Spain lockdown on Day 20 by reducing Ro from 8.0 to 3.0.
        if (day == 20)
        {
            sim.params.Ro = 3.0;
        }

        sim.timeStep(day);

        int countS = 0, countI = 0, countR = 0, countD = 0;
        for (CellState cell : sim.getGridState())
        {
            switch (cell)
            {
            case SUSCEPTIBLE:
                ++countS;
                break;
            case INFECTED:
                ++countI;
                break;
            case RECOVERED:
                ++countR;
                break;
            case DEAD:
                ++countD;
                break;
            }
        }

        const int dailyDeaths = countD - prevDead;
        prevDead = countD;

        std::cout << std::format("{:5d} | {:12d} | {:10d} | {:10d} | {:8d} | {:12d}\n",
                                 day, countS, countI, countR, countD, dailyDeaths);
    }

    return 0;
}
