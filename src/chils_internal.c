#include <omp.h>
#include <limits.h>
#include <stdlib.h>

#include "chils_internal.h"

#define MIN_CORE 512
#define COLLAPSE_DIV 8
#define STAG_ROUNDS 5
#define ELITE_KICK_DIV 100
#define FRESH_KICK_DIV 10
#define REGION_MIN 16
#define CHILS_DBG 0
#define DEFAULT_STEP_TIME 10.0
#define DEFAULT_STEP_COUNT LLONG_MAX

chils *chils_init(graph *g, int p, unsigned int seed)
{
    chils *c = malloc(sizeof(chils));

    c->p = p;
    c->step_time = DEFAULT_STEP_TIME;
    c->step_count = DEFAULT_STEP_COUNT;

    c->cost = 0;
    c->size = 0;
    c->time = 0.0;

    c->LS = malloc(sizeof(local_search *) * p);
    c->LS_core = malloc(sizeof(local_search *) * p);

    c->d_core = malloc(sizeof(graph));
    c->d_core->n = 0;
    c->core0 = 0;
    c->d_core->m = 0;
    c->d_core->V = malloc(sizeof(long long) * (g->n + 1));
    c->d_core->V[0] = 0;
    c->d_core->E = malloc(sizeof(int) * g->V[g->n]);
    c->d_core->W = malloc(sizeof(long long) * g->n);

    c->FM = malloc(sizeof(int) * g->n);
    c->RM = malloc(sizeof(int) * g->n);
    c->A = malloc(sizeof(int) * g->n);

    int nt;
#pragma omp parallel
    {
#pragma omp master
        {
            nt = omp_get_num_threads();
        }
    }

    c->S1 = malloc(sizeof(long long) * nt);
    c->S2 = malloc(sizeof(long long) * nt);

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < p; i++)
        {
            c->LS[i] = local_search_init(g, seed + i);
            c->LS[i]->tournament = PERTURBE_TOURNAMENT;
            c->LS_core[i] = local_search_init(g, seed + p + i);
            c->LS_core[i]->tournament = PERTURBE_TOURNAMENT;
        }
#pragma omp for
        for (int i = 0; i < nt; i++)
        {
            c->S1[i] = 0;
            c->S2[i] = 0;
        }
#pragma omp for
        for (int i = 0; i < g->n; i++)
        {
            c->FM[i] = -1;
            c->RM[i] = -1;
            c->A[i] = 0;
        }
    }

    return c;
}

void chils_free(chils *c)
{
    if (c == NULL)
        return;

    graph_free(c->d_core);

    for (int i = 0; i < c->p; i++)
    {
        local_search_free(c->LS[i]);
        local_search_free(c->LS_core[i]);
    }

    free(c->LS);
    free(c->LS_core);

    free(c->FM);
    free(c->RM);
    free(c->A);
    free(c->S1);
    free(c->S2);

    free(c);
}

static inline int chils_find_overall_best(chils *c)
{
    int best = 0;
    for (int i = 1; i < c->p; i++)
        if (c->LS[i]->cost > c->LS[best]->cost ||
            (c->LS[i]->cost == c->LS[best]->cost && c->LS[i]->time < c->LS[best]->time))
            best = i;
    return best;
}

static inline int chils_find_first_best(chils *c)
{
    int best = 0;
    for (int i = 1; i < c->p; i++)
        if (c->LS[i]->cost > c->LS[best]->cost)
            best = i;
    return best;
}

static inline int chils_find_first_worst(chils *c)
{
    int worst = 0;
    for (int i = 1; i < c->p; i++)
        if (c->LS[i]->cost < c->LS[worst]->cost)
            worst = i;
    return worst;
}

void chils_print(chils *c, long long it, double elapsed)
{
    int best = chils_find_overall_best(c), worst = chils_find_first_worst(c);
    printf("\r%6lld: %12lld (%3d %8.2lf) %12lld (%3d %8.2lf) %8.2lf %9d %12lld",
           it, c->LS[best]->cost, best, c->LS[best]->time,
           c->LS[worst]->cost, worst, c->LS[worst]->time,
           elapsed, c->d_core->n, c->d_core->m);
    fflush(stdout);
}

void chils_update_best(chils *c)
{
    int best = chils_find_overall_best(c);
    c->cost = c->LS[best]->cost;
    c->size = c->LS[best]->size;
    c->time = c->LS[best]->time;
}

void chils_run(graph *g, chils *c, double tl, long long cl, int verbose)
{
    double start = omp_get_wtime();
    double end = omp_get_wtime();
    double elapsed = end - start;

    if (verbose)
    {
        if (cl < LLONG_MAX)
            printf("Running chils for %.2lf seconds or %lld iterations\n", tl, cl);
        else
            printf("Running chils for %.2lf seconds\n", tl);
        printf("%7s %12s (%3s %8s) %12s (%3s %8s) %8s %9s %12s\n", "It.",
               "Best WIS", "id", "time",
               "Worst WIS", "id", "time",
               "time", "D-Core V", "D-Core E");
        chils_print(c, 0, elapsed);
    }

    long long stag_best = 0;
    int stag_rounds = 0;

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < c->p; i++)
        {
            if (c->LS[i]->cost == 0 && i == 0)
                local_search_in_order_solution(g, c->LS[i]);
            else if (c->LS[i]->cost == 0)
                local_search_add_vertex(g, c->LS[i], my_rand_r(&c->LS[i]->seed) % g->n);

            local_search_greedy(g, c->LS[i]);
        }

#pragma omp single
        {
            end = omp_get_wtime();
            elapsed = end - start;
            chils_update_best(c);
            if (verbose)
                chils_print(c, 0, elapsed);
        }

        long long ci = 0;
        while (ci++ < cl && elapsed < tl && keep_running)
        {
            /* Full graph LS */
#pragma omp for
            for (int i = 0; i < c->p; i++)
            {
                double remaining_time = tl - (omp_get_wtime() - start);
                double duration = c->step_time;
                if (remaining_time < duration)
                    duration = remaining_time;
                /* Focus part of the perturbations on the last D-core (contested vertices) */
                c->LS[i]->focus = c->RM;
                c->LS[i]->focus_n = c->d_core->n;
                if (duration > 0.0)
                    local_search_explore(g, c->LS[i], duration, c->step_count, 0);
                c->LS[i]->focus_n = 0;
            }

            /* Mark the D-core */
#pragma omp for nowait
            for (int i = 0; i < g->n; i++)
            {
                int t = 0;
                for (int j = 0; j < c->p; j++)
                    t += c->LS[j]->independent_set[i];
                c->A[i] = t > 0 && t < c->p;
            }

            /* Find the best solution */
            int best = chils_find_first_best(c);

            /* Construct the D-core */
#pragma omp single
            {
                end = omp_get_wtime();
                elapsed = end - start;
                chils_update_best(c);
                if (verbose)
                    chils_print(c, ci, elapsed);
            }

            graph_subgraph_par(g, c->d_core, c->A, c->RM, c->FM, c->S1, c->S2);

#pragma omp single
            {
                if (c->core0 == 0)
                    c->core0 = c->d_core->n;
            }

#pragma omp single
            {
                end = omp_get_wtime();
                elapsed = end - start;
                if (verbose)
                    chils_print(c, ci, elapsed);
            }

            /* D-core LS */
#pragma omp for
            for (int i = 0; i < c->p; i++)
            {
                if (c->d_core->n == 0)
                    continue;

                double remaining_time = tl - (omp_get_wtime() - start);
                double duration = c->step_time * 0.5;
                if (remaining_time < duration)
                    duration = remaining_time;

                if (duration < 0.0)
                    continue;

                local_search_reset(c->d_core, c->LS_core[i]);
                c->LS_core[i]->time_ref = c->LS[i]->time_ref;
                /* offer S1_v7: once the D-core has collapsed below 1/COLLAPSE_DIV of its
                   first-round size, each member starts its core LS from its own random-order
                   construction instead of the shared vertex-order one. */
                if (c->d_core->n * COLLAPSE_DIV < c->core0)
                    local_search_add_vertex(c->d_core, c->LS_core[i],
                                            my_rand_r(&c->LS_core[i]->seed) % c->d_core->n);

                long long ref = 0;
                for (int u = 0; u < c->d_core->n; u++)
                    if (c->LS[i]->independent_set[c->RM[u]])
                        ref += c->d_core->W[u];

                local_search_explore(c->d_core, c->LS_core[i], duration, c->step_count, 0);

                if (ref <= c->LS_core[i]->cost || (i != best && (i % 2) == 0))
                    for (int u = 0; u < c->d_core->n; u++)
                        if (c->LS_core[i]->independent_set[u] && !c->LS[i]->independent_set[c->RM[u]])
                            local_search_add_vertex(g, c->LS[i], c->RM[u]);

                if (ref < c->LS_core[i]->cost)
                    c->LS[i]->time = c->LS_core[i]->time;
            }

            /* Find the best solution after LS on the CHILS core */
            best = chils_find_first_best(c);

#pragma omp barrier
#pragma omp for
            for (int i = 0; i < c->p; i++)
            {
                if (c->d_core->n < MIN_CORE && i != best && (i % 2) == 0)
                {
                    c->LS[i]->tournament = 1;
                    local_search_perturbe(g, c->LS[i]);
                    c->LS[i]->tournament = PERTURBE_TOURNAMENT;
                }
            }

#pragma omp single
            {
                end = omp_get_wtime();
                elapsed = end - start;
                chils_update_best(c);
                if (verbose)
                    chils_print(c, ci, elapsed);

                /* Restart half of the population when the best stagnates */
                if (c->cost > stag_best)
                {
                    stag_best = c->cost;
                    stag_rounds = 0;
                }
                else if (++stag_rounds >= STAG_ROUNDS)
                {
                    stag_rounds = 0;
                    int ob = chils_find_overall_best(c);
                    for (int v = 0; v < g->n; v++)
                        c->A[v] = 0;
                    for (int i = 0; i < c->p; i += 2)
                    {
                        if (i == ob)
                            continue;
                        local_search *l = c->LS[i];
                        double tr = l->time_ref;
                        local_search_reset(g, l);
                        l->time_ref = tr;
                        /* Restart from the elite: copy of the overall best plus a large random kick. */
                        int *B = c->LS[ob]->independent_set;
                        for (int v = 0; v < g->n; v++)
                            if (B[v])
                                local_search_add_vertex(g, l, v);
                        int kick = g->n / ((i % 4 == 2) ? FRESH_KICK_DIV : ELITE_KICK_DIV);
                        if (kick < REGION_MIN)
                            kick = REGION_MIN;
                        if (kick > g->n)
                            kick = g->n;
                        /* Region kick: BFS ball of `kick` vertices around a random vertex,
                           re-randomised by forcing a random maximal independent set inside it.
                           A (core marks) and FM (subgraph map) are free here: both are rebuilt. */
                        int *Q = c->FM, *M = c->A, qn = 0;
                        int r = my_rand_r(&l->seed) % g->n;
                        Q[qn++] = r;
                        M[r] = 1;
                        for (int h = 0; h < qn && qn < kick; h++)
                        {
                            int u = Q[h];
                            for (long long e = g->V[u]; e < g->V[u + 1] && qn < kick; e++)
                            {
                                int v = g->E[e];
                                if (!M[v])
                                {
                                    M[v] = 1;
                                    Q[qn++] = v;
                                }
                            }
                        }
                        for (int k = qn - 1; k > 0; k--)
                        {
                            int x = my_rand_r(&l->seed) % (k + 1);
                            int t = Q[k];
                            Q[k] = Q[x];
                            Q[x] = t;
                        }
                        for (int k = 0; k < qn; k++)
                        {
                            int u = Q[k];
                            if (M[u] != 1)
                                continue;
                            if (!l->independent_set[u])
                                local_search_add_vertex(g, l, u);
                            M[u] = 2;
                            for (long long e = g->V[u]; e < g->V[u + 1]; e++)
                                if (M[g->E[e]] == 1)
                                    M[g->E[e]] = 3;
                        }
                        for (int k = 0; k < qn; k++)
                            M[Q[k]] = 0;
                        local_search_greedy(g, l);
                    }
                }
                if (CHILS_DBG) { int b = chils_find_overall_best(c), w = chils_find_first_worst(c);
                  fprintf(stderr, "DBG %lld %.2f best %lld worst %lld core %d %lld it %lld\n", ci, elapsed, c->LS[b]->cost, c->LS[w]->cost, c->d_core->n, c->d_core->m, c->LS[1]->last_iters); }
            }
        }
    }

    if (verbose)
        printf("\n");
}

void chils_set_solution(graph *g, chils *c, int i, const int *I)
{
    for (int j = 0; j < g->n; j++)
        if (I[j])
            local_search_add_vertex(g, c->LS[i], j);
}

int *chils_get_best_independent_set(chils *c)
{
    int best = chils_find_overall_best(c);

    return c->LS[best]->independent_set;
}
