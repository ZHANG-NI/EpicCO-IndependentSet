#include "local_search.h"

#include <stdlib.h>
#include <limits.h>
#include <assert.h>
#include <omp.h>

#define MAX_GUESS 128
#define MAX_TWO_ONE_DEGREE (1 << 8)
#define AAP_LIMIT (1 << 19)
#define DEFAULT_QUEUE_SIZE 32
#define KICK_STAG 10000
#define KICK_TRIES 4
#define KICK_MAX 3

local_search *local_search_init(graph *g, unsigned int seed)
{
    local_search *ls = malloc(sizeof(local_search));

    ls->independent_set = malloc(sizeof(int) * g->n);

    ls->queue = malloc(sizeof(int) * g->n);
    ls->in_queue = malloc(sizeof(int) * g->n);
    ls->prev_queue = malloc(sizeof(int) * g->n);
    ls->in_prev_queue = malloc(sizeof(int) * g->n);

    ls->max_queue = DEFAULT_QUEUE_SIZE;
    ls->adjacent_weight = malloc(sizeof(long long) * g->n);
    ls->tabu = malloc(sizeof(int) * g->n);
    ls->tightness = malloc(sizeof(int) * g->n);
    ls->temp = malloc(sizeof(int) * g->n * 2);
    ls->mask = malloc(sizeof(int) * g->n);

    ls->log_alloc = g->n;
    ls->log = malloc(sizeof(int) * ls->log_alloc);

    ls->seed = seed;
    ls->tournament = 1;
    ls->focus = NULL;
    ls->focus_n = 0;
    ls->last_iters = 0;

    local_search_reset(g, ls);

    return ls;
}

void local_search_free(local_search *ls)
{
    if (ls == NULL)
        return;

    free(ls->independent_set);

    free(ls->queue);
    free(ls->in_queue);
    free(ls->prev_queue);
    free(ls->in_prev_queue);

    free(ls->adjacent_weight);
    free(ls->tabu);
    free(ls->tightness);
    free(ls->temp);
    free(ls->mask);

    free(ls->log);

    free(ls);
}

void local_search_reset(graph *g, local_search *ls)
{
    ls->cost = 0;
    ls->size = 0;
    ls->time = 0.0;
    ls->time_ref = omp_get_wtime();

    ls->queue_count = g->n;

    ls->log_count = 0;
    ls->log_enabled = 0;
    ls->greedy_filter = 0;
    ls->stag = 0;

    for (int u = 0; u < g->n; u++)
    {
        ls->independent_set[u] = 0;
        ls->queue[u] = u;
        ls->in_queue[u] = 1;
        ls->prev_queue[u] = 0;
        ls->in_prev_queue[u] = 0;

        ls->adjacent_weight[u] = 0;
        ls->tabu[u] = 0;
        ls->tightness[u] = 0;
        ls->temp[u] = 0;
        ls->mask[u] = 0;
    }
}

static inline void local_search_shuffle(int *list, int n, unsigned int *seed)
{
    for (int i = 0; i < n - 1; i++)
    {
        int j = i + (my_rand_r(seed) % (n - i));
        int t = list[j];
        list[j] = list[i];
        list[i] = t;
    }
}

void local_search_in_order_solution(graph *g, local_search *ls)
{
    for (int u = 0; u < g->n; u++)
    {
        if (!ls->tabu[u] && ls->adjacent_weight[u] < g->W[u])
            local_search_add_vertex(g, ls, u);
    }
}

void local_search_add_vertex(graph *g, local_search *ls, int u)
{
    assert(!ls->independent_set[u] && !ls->tabu[u]);

    ls->independent_set[u] = 1;
    ls->cost += g->W[u];
    ls->size += 1;

    if (!ls->in_queue[u])
    {
        ls->in_queue[u] = 1;
        ls->queue[ls->queue_count] = u;
        ls->queue_count++;
    }

    for (long long i = g->V[u]; i < g->V[u + 1]; i++)
    {
        int v = g->E[i];
        if (ls->independent_set[v])
            local_search_remove_vertex(g, ls, v);

        ls->adjacent_weight[v] += g->W[u];
        ls->tightness[v]++;
    }
}

void local_search_remove_vertex(graph *g, local_search *ls, int u)
{
    assert(ls->independent_set[u] && !ls->tabu[u]);

    if (ls->log_enabled)
    {
        if (ls->log_count >= ls->log_alloc)
        {
            ls->log_alloc *= 2;
            ls->log = realloc(ls->log, sizeof(int) * ls->log_alloc);
        }
        ls->log[ls->log_count++] = u;
    }

    ls->independent_set[u] = 0;
    ls->cost -= g->W[u];
    ls->size -= 1;

    if (!ls->in_queue[u])
    {
        ls->in_queue[u] = 1;
        ls->queue[ls->queue_count] = u;
        ls->queue_count++;
    }

    for (long long i = g->V[u]; i < g->V[u + 1]; i++)
    {
        int v = g->E[i];
        ls->adjacent_weight[v] -= g->W[u];
        ls->tightness[v]--;

        /* During greedy, only neighbours that greedy can act on are queued:
           insertable ones (adjacent weight below own weight) or tight ones when AAP is on.
           Any later improvement of v comes from another removal, which re-checks. */
        if (!ls->in_queue[v] && !ls->tabu[v] &&
            (!ls->greedy_filter || ls->adjacent_weight[v] < g->W[v] ||
             (ls->tightness[v] == 1 && g->m < AAP_LIMIT)))
        {
            ls->in_queue[v] = 1;
            ls->queue[ls->queue_count] = v;
            ls->queue_count++;
        }
    }
}

void local_search_lock_vertex(graph *g, local_search *ls, int u)
{
    ls->tabu[u]++;
    if (ls->independent_set[u])
    {
        for (long long i = g->V[u]; i < g->V[u + 1]; i++)
            ls->tabu[g->E[i]]++;
    }
}

void local_search_unlock_vertex(graph *g, local_search *ls, int u)
{
    ls->tabu[u]--;
    if (!ls->in_queue[u] && !ls->tabu[u])
    {
        ls->in_queue[u] = 1;
        ls->queue[ls->queue_count] = u;
        ls->queue_count++;
    }
    if (!ls->independent_set[u])
        return;

    for (long long i = g->V[u]; i < g->V[u + 1]; i++)
    {
        int v = g->E[i];
        ls->tabu[v]--;

        if (!ls->in_queue[v] && !ls->tabu[v])
        {
            ls->in_queue[v] = 1;
            ls->queue[ls->queue_count] = v;
            ls->queue_count++;
        }
    }
}

void local_search_two_one(graph *g, local_search *ls, int u)
{
    assert(ls->independent_set[u] && !ls->tabu[u]);

    int adjacent_count = 0;
    long long max_w = LLONG_MIN;
    for (long long i = g->V[u]; i < g->V[u + 1]; i++)
    {
        int v = g->E[i];
        if (ls->tightness[v] == 1 && !ls->tabu[v])
        {
            ls->temp[adjacent_count++] = v;
            if (g->W[v] > max_w)
                max_w = g->W[v];
        }
    }

    if (adjacent_count < 2)
        return;

    int b1 = -1, b2 = -1;
    long long best = LLONG_MIN;
    for (int i = 0; i < adjacent_count; i++)
    {
        int v = ls->temp[i];

        /* No partner can make the swap improving: skip the neighbour scan of v. */
        if (g->W[v] + max_w <= g->W[u])
            continue;

        int i1 = 0;
        long long i2 = g->V[v];
        while (i1 < adjacent_count && i2 < g->V[v + 1])
        {
            int w1 = ls->temp[i1], w2 = g->E[i2];
            if (w1 > w2)
                i2++;
            else if (w1 == w2)
                i1++, i2++;
            else if (w1 == v)
                i1++;
            else // (w1 < w2) Found 2-1 swap
            {
                i1++;
                if (g->W[w1] + g->W[v] <= g->W[u])
                    continue;

                long long gain = (my_rand_r(&ls->seed) % (1 << 30)) - (1 << 29);
                long long diff = (g->W[w1] + g->W[v]) - g->W[u];

                if (diff + gain > best)
                {
                    best = diff + gain;
                    b1 = v;
                    b2 = w1;
                }
            }
        }
    }
    if (b1 >= 0)
    {
        local_search_add_vertex(g, ls, b1);
        local_search_add_vertex(g, ls, b2);
    }
}

void local_search_aap(graph *g, local_search *ls, int u, int imp)
{
    assert(ls->independent_set[u] || ls->tightness[u] == 1);

    int current = -1, candidate_size = 0;
    ls->temp[candidate_size++] = u;

    if (ls->independent_set[u])
    {
        current = u;
    }
    else
    {
        for (long long i = g->V[u]; i < g->V[u + 1]; i++)
        {
            int v = g->E[i];
            if (ls->independent_set[v])
            {
                current = v;
                ls->temp[candidate_size++] = v;
                break;
            }
        }

        if (current < 0 || ls->tabu[current])
            return;
        ls->mask[u] = 1;
    }
    ls->mask[current] = 2;

    int found = 1;
    while (found)
    {
        found = 0;
        long long best = INT_MIN;
        int to_add, to_remove;

        for (long long i = g->V[current]; i < g->V[current + 1]; i++)
        {
            int v = g->E[i];
            if (ls->tightness[v] > 2 || ls->mask[v] == 1 || ls->tabu[v] || g->W[v] < 1)
                continue;

            int valid = 1, next = current;

            for (long long j = g->V[v]; j < g->V[v + 1] && valid; j++)
            {
                int w = g->E[j];
                if (w == current)
                    continue;

                if (ls->mask[w] == 1)
                    valid = 0;
                else if (ls->independent_set[w])
                    next = w;
            }

            long long gain;
            if (!imp)
                gain = (my_rand_r(&ls->seed) % (1 << 30)) - (1 << 29);
            else
                gain = (my_rand_r(&ls->seed) % (2 * g->W[v])) - (g->W[v]);

            long long change = g->W[v];
            if (next >= 0 && ls->mask[next] != 2)
                change -= g->W[next];

            if (valid && !ls->tabu[next] && change + gain > best)
            {
                to_add = v;
                to_remove = next;
                best = change + gain;
                found = 1;
            }
        }

        if (found)
        {
            ls->temp[candidate_size++] = to_add;
            ls->mask[to_add] = 1;
            if (ls->mask[to_remove] != 2)
            {
                ls->temp[candidate_size++] = to_remove;
                ls->mask[to_remove] = 2;
            }
            current = to_remove;
        }
    }

    long long diff = 0, best = LLONG_MIN;
    int best_position = 0;
    int to_add = 0;

    for (int i = 0; i < candidate_size; i++)
    {
        int v = ls->temp[i];
        if (ls->independent_set[v] && ls->mask[v] == 2)
        {
            diff -= g->W[v];
            ls->mask[v] = 1;
        }
        else if (!ls->independent_set[v])
        {
            diff += g->W[v];
            ls->temp[(long long)g->n + to_add++] = v;
        }

        if (ls->independent_set[v] && diff > best)
        {
            best = diff;
            best_position = to_add;
        }
    }

    if (best <= 0)
        best_position = to_add;

    if (best > 0 || !imp)
    {
        for (int i = 0; i < best_position; i++)
        {
            int v = ls->temp[(long long)g->n + i];
            local_search_add_vertex(g, ls, v);
        }
    }

    for (int i = 0; i < candidate_size; i++)
        ls->mask[ls->temp[i]] = 0;
}

static inline void local_search_swap(int **a, int **b)
{
    int *t = *a;
    *a = *b;
    *b = t;
}

void local_search_greedy(graph *g, local_search *ls)
{
    /* Drop queued vertices on which greedy cannot act (not in the solution, not
       insertable, not tight for AAP) before shuffling; the order of the rest stays uniform. */
    {
        int k = 0, aap = g->m < AAP_LIMIT;
        for (int i = 0; i < ls->queue_count; i++)
        {
            int u = ls->queue[i];
            if (!ls->tabu[u] && (ls->independent_set[u] || ls->adjacent_weight[u] < g->W[u] ||
                                 (aap && ls->tightness[u] == 1)))
                ls->queue[k++] = u;
            else
                ls->in_queue[u] = 0;
        }
        ls->queue_count = k;
    }
    local_search_shuffle(ls->queue, ls->queue_count, &ls->seed);

    int n = ls->queue_count;
    ls->queue_count = 0;
    ls->greedy_filter = 1;
    while (n > 0)
    {
        local_search_swap(&ls->queue, &ls->prev_queue);
        local_search_swap(&ls->in_queue, &ls->in_prev_queue);

        for (int i = 0; i < n; i++)
        {
            int u = ls->prev_queue[i];
            ls->in_prev_queue[u] = 0;

            if (ls->tabu[u])
                continue;

            if (!ls->independent_set[u] && ls->adjacent_weight[u] < g->W[u])
                local_search_add_vertex(g, ls, u);
            else if (ls->independent_set[u])
                local_search_two_one(g, ls, u);

            if (ls->tightness[u] == 1 && g->m < AAP_LIMIT)
                local_search_aap(g, ls, u, 1);
        }

        local_search_shuffle(ls->queue, ls->queue_count, &ls->seed);

        n = ls->queue_count;
        ls->queue_count = 0;
    }
    ls->greedy_filter = 0;
}

/* Random vertex: from the focus set (vertices on which the population disagrees)
   when focused is set, else uniformly from the graph. */
static inline int local_search_draw(graph *g, local_search *ls, int focused)
{
    if (focused)
        return ls->focus[my_rand_r(&ls->seed) % ls->focus_n];
    return my_rand_r(&ls->seed) % g->n;
}

void local_search_perturbe(graph *g, local_search *ls)
{
    /* Focus only when the uniform perturbations of one step cover a small part of the
       graph (last call's iterations < n/8): then most would miss the contested region. */
    int focused = ls->focus_n > 0 && ls->last_iters * FOCUS_COVER_DIV < g->n &&
                  (my_rand_r(&ls->seed) & 3) == 0;
    int u = local_search_draw(g, ls, focused);
    int q = 0;
    while (q++ < MAX_GUESS && ls->tabu[u])
        u = local_search_draw(g, ls, focused);

    if (ls->tabu[u])
        return;

    long long best = ls->cost;

    if (g->m < AAP_LIMIT && (ls->independent_set[u] || ls->tightness[u] == 1))
    {
        local_search_aap(g, ls, u, ls->tightness[u] == 1);
    }
    else
    {
        /* Tournament: among a few random candidates outside the solution,
           force the one whose insertion loses the least weight. */
        if (!ls->independent_set[u])
        {
            long long bl = ls->adjacent_weight[u] - g->W[u];
            for (int k = 1; k < ls->tournament; k++)
            {
                int v = local_search_draw(g, ls, focused);
                if (ls->tabu[v] || ls->independent_set[v])
                    continue;
                if (g->m < AAP_LIMIT && ls->tightness[v] == 1)
                    continue;
                long long l = ls->adjacent_weight[v] - g->W[v];
                if (l < bl)
                {
                    bl = l;
                    u = v;
                }
            }
        }

        if (ls->independent_set[u])
            local_search_remove_vertex(g, ls, u);
        else
            local_search_add_vertex(g, ls, u);

        local_search_lock_vertex(g, ls, u);

        /* Local multi-vertex kick once the ILS has stagnated: also force a vertex
           at distance 2 of u (min-loss among a few samples), locked like u. */
        int ys[KICK_MAX], ny = 0;
        if (ls->stag > KICK_STAG && ls->independent_set[u] && g->V[u + 1] > g->V[u] &&
            (my_rand_r(&ls->seed) & 1))
        {
            /* Kick size grows by one each time the stagnation quadruples. */
            int kmax = 0;
            for (long long t = KICK_STAG; ls->stag > t && kmax < KICK_MAX; t *= 4)
                kmax++;
            int kk = 1 + my_rand_r(&ls->seed) % kmax;
            long long du = g->V[u + 1] - g->V[u];
            for (int e = 0; e < kk; e++)
            {
                int y = -1;
                long long bl = LLONG_MAX;
                for (int k = 0; k < KICK_TRIES; k++)
                {
                    int x = g->E[g->V[u] + my_rand_r(&ls->seed) % du];
                    long long dx = g->V[x + 1] - g->V[x];
                    int z = g->E[g->V[x] + my_rand_r(&ls->seed) % dx];
                    if (z == u || ls->tabu[z] || ls->independent_set[z])
                        continue;
                    long long l = ls->adjacent_weight[z] - g->W[z];
                    if (l < bl)
                    {
                        bl = l;
                        y = z;
                    }
                }
                if (y < 0)
                    continue;
                local_search_add_vertex(g, ls, y);
                local_search_lock_vertex(g, ls, y);
                ys[ny++] = y;
            }
        }

        for (int i = 0; i < MAX_GUESS &&
                        ls->queue_count > 0 &&
                        ls->queue_count < ls->max_queue &&
                        ls->cost <= best;
             i++)
        {
            int v = ls->queue[my_rand_r(&ls->seed) % ls->queue_count];
            q = 0;
            while (q++ < MAX_GUESS && ls->tabu[v])
                v = ls->queue[my_rand_r(&ls->seed) % ls->queue_count];

            if (ls->tabu[v])
                continue;

            if (ls->independent_set[v])
                local_search_remove_vertex(g, ls, v);
            else
                local_search_add_vertex(g, ls, v);
        }

        while (ny > 0)
            local_search_unlock_vertex(g, ls, ys[--ny]);
        local_search_unlock_vertex(g, ls, u);
    }
}

void local_search_explore(graph *g, local_search *ls, double tl, long long il, int verbose)
{
    long long best = ls->cost, c = 0;

    if (verbose)
    {
        if (il < LLONG_MAX)
            printf("Running baseline local search for %.2lf seconds or %lld iterations\n", tl, il);
        else
            printf("Running baseline local search for %.2lf seconds\n", tl);
        printf("%11s %12s %8s\n", "It.", "WIS", "Time");
        printf("\r%10lld: %12lld %8.2lf", 0ll, ls->cost, 0.0);
        fflush(stdout);
    }

    double start = omp_get_wtime();

    ls->log_enabled = 0;
    if (ls->cost == 0)
        local_search_in_order_solution(g, ls);
    local_search_greedy(g, ls);

    if (ls->cost > best)
    {
        best = ls->cost;
        ls->time = omp_get_wtime() - ls->time_ref;
        if (verbose)
        {
            printf("\r%10lld: %12lld %8.2lf", 0ll, ls->cost, ls->time);
            fflush(stdout);
        }
    }

    while (c < il)
    {
        if ((c++ & ((1 << 7) - 1)) == 0)
        {
            // c = 0;
            if (omp_get_wtime() - start > tl || !keep_running)
                break;

            if (verbose)
            {
                printf("\r%10lld: %12lld %8.2lf", c, ls->cost, ls->time);
                fflush(stdout);
            }
        }

        ls->log_count = 0;
        ls->log_enabled = 1;

        local_search_perturbe(g, ls);

        local_search_greedy(g, ls);

        if (ls->cost > best)
        {
            ls->stag = 0;
            best = ls->cost;
            ls->time = omp_get_wtime() - ls->time_ref;
            ls->log_count = 0;
            if (verbose)
            {
                printf("\r%10lld: %12lld %8.2lf", c, ls->cost, ls->time);
                fflush(stdout);
            }
        }
        else
            ls->stag++;
        if (ls->cost < best)
        {
            local_search_unwind(g, ls, 0);
        }
    }
    ls->last_iters = c;
    if (verbose)
        printf("\n");
}

void local_search_unwind(graph *g, local_search *ls, int t)
{
    ls->log_enabled = 0;
    while (ls->log_count > t)
    {
        ls->log_count--;
        int u = ls->log[ls->log_count];

        if (!ls->independent_set[u])
            local_search_add_vertex(g, ls, u);
    }
}
