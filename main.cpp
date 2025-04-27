#include <iostream>
#include <vector>
#include <numeric>   // Dla std::accumulate, std::max_element
#include <algorithm> // Dla std::sort, std::min_element, std::max_element, std::min, std::max
#include <limits>    // Dla std::numeric_limits
#include <functional> // Dla std::greater, std::function
#include <random>   // Do generowania liczb losowych
#include <chrono>   // Do pomiaru czasu
#include <iomanip>  // Do formatowania wyjścia
#include <string>   // Dla std::string
#include <utility>  // Dla std::pair
#include <cmath>    // Dla std::floor
#include <stdexcept> // Dla std::invalid_argument, std::bad_alloc
#include "sstream" // Dla std::stringstream
// Używamy long long dla sum czasów, aby uniknąć przepełnienia
using TimeType = long long;

// --- Funkcje algorytmów (bez zmian) ---

/**
 * @brief Znajduje indeks maszyny z najmniejszym aktualnym obciążeniem.
 */
int find_least_loaded_machine(const std::vector<TimeType>& machine_loads) {
    if (machine_loads.empty()) return -1;
    int min_idx = 0;
    TimeType min_load = machine_loads[0];
    for (int i = 1; i < machine_loads.size(); ++i) {
        if (machine_loads[i] < min_load) {
            min_load = machine_loads[i];
            min_idx = i;
        }
    }
    return min_idx;
}

/**
 * @brief Implementacja algorytmu List Scheduling (LSA).
 */
TimeType list_scheduling(int m, const std::vector<int>& tasks, std::vector<TimeType>& machine_loads) {
    if (m <= 0) return 0;
    machine_loads.assign(m, 0LL);
    if (tasks.empty()) return 0;

    for (int task_duration : tasks) {
        int target_machine_idx = find_least_loaded_machine(machine_loads);
        if (target_machine_idx != -1) {
            machine_loads[target_machine_idx] += task_duration;
        } else {
            return -1; // Błąd
        }
    }

    return *std::max_element(machine_loads.begin(), machine_loads.end());
}

/**
 * @brief Implementacja algorytmu Longest Processing Time First (LPT).
 */
TimeType longest_processing_time(int m, const std::vector<int>& tasks, std::vector<TimeType>& machine_loads) {
    if (m <= 0) return 0;
    machine_loads.assign(m, 0LL);
    if (tasks.empty()) return 0;

    std::vector<int> sorted_tasks = tasks;
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    for (int task_duration : sorted_tasks) {
        int target_machine_idx = find_least_loaded_machine(machine_loads);
        if (target_machine_idx != -1) {
            machine_loads[target_machine_idx] += task_duration;
        } else {
            return -1; // Błąd
        }
    }

    return *std::max_element(machine_loads.begin(), machine_loads.end());
}

/**
 * @brief Implementacja algorytmu Programowania Dynamicznego dla P2||Cmax.
 */
TimeType dynamic_programming_p2(int m, const std::vector<int>& tasks, std::vector<TimeType>& machine_loads) {
    if (m != 2) {
        // Zamiast zwracać LPT, rzucimy wyjątek, bo DP jest specyficzne dla m=2
        throw std::invalid_argument("Algorytm DP zaimplementowany tylko dla m=2.");
    }
    machine_loads.assign(m, 0LL);
    if (tasks.empty()) {
        return 0;
    }

    TimeType total_sum = 0;
    for (int task_duration : tasks) {
        if (task_duration < 0) {
            throw std::invalid_argument("Czas zadania nie moze byc ujemny.");
        }
        if (total_sum > std::numeric_limits<TimeType>::max() - task_duration) {
            throw std::overflow_error("Przepełnienie TimeType podczas sumowania czasow zadan.");
        }
        total_sum += task_duration;
    }

    TimeType target_sum = total_sum / 2;

    // --- Optymalizacja dla bardzo dużego target_sum ---
    // Jeśli target_sum jest zbyt duże, alokacja dp może się nie udać lub być zbyt wolna.
    // Można ustawić jakiś limit (np. 100 milionów) i zwrócić błąd, jeśli jest przekroczony.
    const TimeType DP_SUM_LIMIT = 100000000; // Przykładowy limit
    if (target_sum > DP_SUM_LIMIT) {
        throw std::runtime_error("Suma docelowa (S/2) jest zbyt duza dla DP: " + std::to_string(target_sum));
    }
    // --- Koniec optymalizacji ---

    std::vector<bool> dp;
    try {
        // +1 bo potrzebujemy indeksów od 0 do target_sum włącznie
        dp.resize(static_cast<size_t>(target_sum) + 1, false);
    } catch (const std::bad_alloc& e) {
        // Zwracamy bardziej konkretny błąd
        throw std::runtime_error("Nie mozna zaalokowac pamieci dla DP (rozmiar: "
                                 + std::to_string(target_sum + 1) + ")");
    }
    dp[0] = true;

    TimeType current_max_reachable = 0;
    for (int task_duration : tasks) {
        if (task_duration <= 0) continue;
        TimeType task_ll = static_cast<TimeType>(task_duration); // Konwersja dla bezpieczeństwa

        // Iterujemy od tyłu, aby nie użyć tego samego zadania wielokrotnie w jednym kroku
        for (TimeType k = std::min(current_max_reachable + task_ll, target_sum); k >= task_ll; --k) {
            if (dp[static_cast<size_t>(k - task_ll)]) {
                dp[static_cast<size_t>(k)] = true;
                // Zaktualizuj current_max_reachable jeśli znaleziono nową osiągalną sumę
                current_max_reachable = std::max(current_max_reachable, k);
            }
        }
        // Optymalizacja: jeśli osiągnięto cel, można by przerwać, ale szukanie best_sum_m1 i tak to znajdzie.
        if (dp[static_cast<size_t>(target_sum)]) {
            current_max_reachable = target_sum;
        }
    }

    TimeType best_sum_m1 = 0;
    for (TimeType k = target_sum; k >= 0; --k) {
        if (dp[static_cast<size_t>(k)]) {
            best_sum_m1 = k;
            break;
        }
    }

    TimeType load_m1 = best_sum_m1;
    TimeType load_m2 = total_sum - best_sum_m1;
    TimeType cmax = std::max(load_m1, load_m2);

    machine_loads[0] = load_m1;
    machine_loads[1] = load_m2;

    return cmax;
}


/**
 * @brief Generuje wektor n zadań z losowymi czasami wykonania z przedziału [min_p, max_p].
 */
std::vector<int> generate_tasks(int n, int min_p, int max_p) {
    if (n < 0 || min_p < 0 || max_p < min_p) {
        throw std::invalid_argument("Nieprawidlowe argumenty dla generate_tasks");
    }
    // Używamy wątkowo-lokalnego generatora dla lepszej losowości w wielowątkowych scenariuszach (chociaż tutaj nie ma)
    thread_local static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> distrib(min_p, max_p);

    std::vector<int> tasks(n);
    for (int i = 0; i < n; ++i) {
        tasks[i] = distrib(gen);
    }
    return tasks;
}

// --- Funkcja pomocnicza do pomiaru czasu ---
using SchedulingAlgorithm = std::function<TimeType(int, const std::vector<int>&, std::vector<TimeType>&)>;

struct AlgoResult {
    TimeType cmax = -1;
    long long duration_us = -1;
    bool success = false;
    std::string error_msg = "";
};

AlgoResult measure_algorithm(
        SchedulingAlgorithm algo,
        int m,
        const std::vector<int>& tasks,
        std::vector<TimeType>& machine_loads // Przekazywane przez referencję (wynik nie jest kluczowy dla pomiarów)
) {
    AlgoResult result;
    std::vector<TimeType> current_loads; // Lokalna kopia dla tego algorytmu

    try {
        auto start = std::chrono::high_resolution_clock::now();
        result.cmax = algo(m, tasks, current_loads);
        auto stop = std::chrono::high_resolution_clock::now();

        // Algorytm zakończył się bez wyjątku
        if (result.cmax >= 0) { // Dodatkowe sprawdzenie, czy nie zwrócił błędu
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            result.duration_us = duration.count();
            result.success = true;
        } else {
            result.error_msg = "Algorytm zwrocil blad (Cmax < 0)";
            result.success = false;
        }

    } catch (const std::exception& e) {
        result.error_msg = "Wyjatek: " + std::string(e.what());
        result.success = false;
    } catch (...) {
        result.error_msg = "Nieznany wyjatek";
        result.success = false;
    }

    // Mimo że 'machine_loads' jest przekazywane przez ref, nie modyfikujemy go na zewnątrz
    // Chyba że byłoby to potrzebne, wtedy: if(result.success) machine_loads = current_loads;

    return result;
}


int main() {
    const int NUM_MACHINES = 2;
    const int NUM_RUNS_PER_CONFIG = 10; // Liczba powtórzeń dla uśrednienia wyników

    struct TestConfig {
        int n;
        int min_p;
        int max_p;
        std::string description() const {
            return "n=" + std::to_string(n) + " p=[" + std::to_string(min_p) + "," + std::to_string(max_p) + "]";
        }
        std::string range_str() const {
            // Zapewnia stałą szerokość dla zakresu
            std::stringstream ss;
            ss << "[" << std::setw(3) << std::right << min_p << "-"
               << std::setw(3) << std::right << max_p << "]";
            return ss.str();
        }
    };

    std::vector<TestConfig> configs = {
            {10, 1, 10},
            {10, 10, 20},
            {20, 1, 10},
            {20, 10, 20},
            {20, 50, 100},
            {50, 1, 10},
            {50, 10, 20},
            {50, 50, 100},
            // Test DP z potencjalnie dużym S
            {30, 500, 1000}, // n*max_p ~ 30k, S/2 ~ 15k * 30 = 450k -> OK
            // Test DP z potencjalnym błędem (duże S) - odkomentuj, aby zobaczyć błąd
            // {50, 100000, 200000} // S/2 ~ 150k * 50 = 7.5M -> OK
            // {100, 100000, 200000} // S/2 ~ 150k * 100 = 15M -> OK
            // {200, 1000000, 2000000} // S/2 ~ 1.5M * 200 = 300M -> Przekroczy limit DP_SUM_LIMIT
    };

    // --- Definicje Szerokości Kolumn ---
    const int W_N = 7;
    const int W_RANGE = 12;
    const int W_CMAX = 14;
    const int W_TIME = 16;
    const int TOTAL_WIDTH = W_N + W_RANGE + 3 * (W_CMAX + W_TIME);

    // --- Nagłówek Tabeli ---
    std::cout << "\n--- Tabela Wynikow Eksperymentow (P2||Cmax) ---\n";
    std::cout << "Liczba powtorzen na konfiguracje: " << NUM_RUNS_PER_CONFIG << "\n\n";

    std::cout << std::left
              << std::setw(W_N) << "n"
              << std::setw(W_RANGE) << "p_j range"
              << std::setw(W_CMAX) << "Avg LSA Cmax"
              << std::setw(W_TIME) << "Avg LSA Time(us)"
              << std::setw(W_CMAX) << "Avg LPT Cmax"
              << std::setw(W_TIME) << "Avg LPT Time(us)"
              << std::setw(W_CMAX) << "Avg DP Cmax"
              << std::setw(W_TIME) << "Avg DP Time(us)"
              << std::endl;
    std::cout << std::string(TOTAL_WIDTH, '-') << std::endl; // Linia oddzielająca

    // --- Pętla Testująca ---
    for (const auto& config : configs) {
        // Akumulatory wyników
        double total_cmax_lsa = 0, total_cmax_lpt = 0, total_cmax_dp = 0;
        double total_duration_lsa = 0, total_duration_lpt = 0, total_duration_dp = 0;
        int runs_lsa_ok = 0, runs_lpt_ok = 0, runs_dp_ok = 0;
        std::string first_dp_error = ""; // Zapisz pierwszy błąd DP dla tej konfiguracji

        for (int run = 0; run < NUM_RUNS_PER_CONFIG; ++run) {
            std::vector<int> tasks;
            try {
                tasks = generate_tasks(config.n, config.min_p, config.max_p);
            } catch (const std::exception& e) {
                std::cerr << "KRYTYCZNY BLAD: Nie udalo sie wygenerowac zadan dla "
                          << config.description() << ": " << e.what() << std::endl;
                // Jeśli generowanie zawodzi, nie ma sensu kontynuować dla tej konfiguracji
                runs_lsa_ok = runs_lpt_ok = runs_dp_ok = -1; // Oznacz jako błąd krytyczny
                break; // Przerwij pętlę wewnętrzną
            }

            std::vector<TimeType> loads; // Wektor pomocniczy

            // Pomiar dla LSA
            AlgoResult res_lsa = measure_algorithm(list_scheduling, NUM_MACHINES, tasks, loads);
            if (res_lsa.success) {
                total_cmax_lsa += res_lsa.cmax;
                total_duration_lsa += res_lsa.duration_us;
                runs_lsa_ok++;
            } // Błędy LSA są mało prawdopodobne, można dodać logowanie 'else'

            // Pomiar dla LPT
            AlgoResult res_lpt = measure_algorithm(longest_processing_time, NUM_MACHINES, tasks, loads);
            if (res_lpt.success) {
                total_cmax_lpt += res_lpt.cmax;
                total_duration_lpt += res_lpt.duration_us;
                runs_lpt_ok++;
            } // Błędy LPT też mało prawdopodobne

            // Pomiar dla DP
            AlgoResult res_dp = measure_algorithm(dynamic_programming_p2, NUM_MACHINES, tasks, loads);
            if (res_dp.success) {
                total_cmax_dp += res_dp.cmax;
                total_duration_dp += res_dp.duration_us;
                runs_dp_ok++;
            } else {
                // Zapisz tylko pierwszy napotkany błąd DP dla danej konfiguracji
                if (first_dp_error.empty()) {
                    first_dp_error = res_dp.error_msg;
                }
            }
        } // Koniec pętli wewnętrznej (runs)

        // --- Wypisanie Wyników dla Konfiguracji ---

        // Wypisz podstawowe informacje
        std::cout << std::left
                  << std::setw(W_N) << config.n
                  << std::setw(W_RANGE) << config.range_str();

        // Funkcja pomocnicza do wypisywania średnich lub "N/A" / "ERROR"
        auto print_avg_result = [&](double total_val, double total_time, int ok_runs, int width_val, int width_time) {
            if (ok_runs > 0) {
                std::cout << std::fixed << std::setprecision(1) // Mniej miejsc po przecinku dla Cmax
                          << std::setw(width_val) << (total_val / ok_runs)
                          << std::fixed << std::setprecision(2) // Więcej dla czasu
                          << std::setw(width_time) << (total_time / ok_runs);
            } else if (ok_runs == 0 && !first_dp_error.empty() && &total_val == &total_cmax_dp) {
                // Specjalny przypadek dla DP: jeśli 0 udanych przebiegów z powodu błędów
                std::cout << std::setw(width_val) << "ERROR"
                          << std::setw(width_time) << "ERROR";
            }
            else if (ok_runs == -1) { // Błąd krytyczny (np. generowania zadań)
                std::cout << std::setw(width_val) << "CRITICAL"
                          << std::setw(width_time) << "CRITICAL";
            }
            else { // 0 udanych przebiegów, brak zapisanego błędu
                std::cout << std::setw(width_val) << "N/A"
                          << std::setw(width_time) << "N/A";
            }
        };

        // Wypisz wyniki dla LSA, LPT, DP
        print_avg_result(total_cmax_lsa, total_duration_lsa, runs_lsa_ok, W_CMAX, W_TIME);
        print_avg_result(total_cmax_lpt, total_duration_lpt, runs_lpt_ok, W_CMAX, W_TIME);
        print_avg_result(total_cmax_dp, total_duration_dp, runs_dp_ok, W_CMAX, W_TIME);

        std::cout << std::endl;

        // Jeśli wystąpił błąd DP, wypisz go pod wierszem wyników
        if (!first_dp_error.empty() && runs_dp_ok < NUM_RUNS_PER_CONFIG) {
            std::cout << std::string(W_N + W_RANGE, ' ') // Wcięcie
                      << "[DP Info: " << runs_dp_ok << "/" << NUM_RUNS_PER_CONFIG << " OK. First error: " << first_dp_error << "]"
                      << std::endl;
        }


    } // Koniec pętli zewnętrznej (configs)

    std::cout << std::string(TOTAL_WIDTH, '-') << std::endl; // Linia końcowa
    std::cout << "Koniec eksperymentow.\n" << std::endl;

    return 0;
}