#include <iostream> // Do obsługi wejścia/wyjścia (cout)
#include <vector>   // Do używania dynamicznych tablic (vector)
#include <numeric>  // Dla std::accumulate (sumowanie), std::max_element
#include <algorithm>// Dla std::sort, std::min_element, std::max_element, std::min, std::max
#include <limits>   // Dla std::numeric_limits (sprawdzanie zakresów typów)
#include <functional>// Dla std::greater (sortowanie malejąco), std::function
#include <random>   // Do generowania liczb losowych
#include <chrono>   // Do pomiaru czasu
#include <iomanip>  // Do formatowania wyjścia (setw, setprecision, left, right, fixed)
#include <string>   // Do używania std::string
#include <utility>  // Dla std::pair (nieużywane bezpośrednio)
#include <cmath>    // Dla std::floor (chociaż nie jest krytyczne)
#include <stdexcept>// Do obsługi standardowych wyjątków (np. invalid_argument)
#include <sstream>  // Dla std::stringstream (konwersja liczb na string, formatowanie)

// --- Definicje Struktur Pomocniczych ---

// Struktura przechowująca wynik pojedynczego uruchomienia algorytmu
struct AlgorithmRunResult {
    long long cmax = -1;          // Wynik Cmax (-1 oznacza błąd)
    long long duration_us = -1;  // Czas wykonania w mikrosekundach (-1 oznacza błąd)
    bool success = false;        // Czy wykonanie się powiodło?
    std::string error_message = ""; // Komunikat błędu, jeśli wystąpił
};

// Struktura opisująca jedną konfigurację testową (rozmiar instancji)
struct InstanceConfig {
    int num_tasks;      // Liczba zadań (n)
    int min_task_time;  // Minimalny czas zadania (min_p)
    int max_task_time;  // Maksymalny czas zadania (max_p)

    // Funkcja pomocnicza do opisu konfiguracji
    std::string description() const {
        return "n=" + std::to_string(num_tasks)
               + " p=[" + std::to_string(min_task_time)
               + "," + std::to_string(max_task_time) + "]";
    }
    // Funkcja pomocnicza do sformatowania zakresu czasów dla tabeli
    std::string range_string_formatted() const {
        std::stringstream ss;
        // Wyrównanie do prawej, szerokość 3 cyfry dla każdej liczby w zakresie
        ss << "[" << std::setw(3) << std::right << min_task_time << "-"
           << std::setw(3) << std::right << max_task_time << "]";
        return ss.str();
    }
};

// --- Implementacje Algorytmów i Funkcji Pomocniczych ---

/**
 * @brief Znajduje indeks maszyny z najmniejszym aktualnym obciążeniem.
 */
int find_least_loaded_machine(const std::vector<long long>& machine_loads) {
    if (machine_loads.empty()) {
        return -1; // Błąd: brak maszyn
    }
    int min_index = 0;
    long long min_load = machine_loads[0];
    for (size_t i = 1; i < machine_loads.size(); ++i) {
        if (machine_loads[i] < min_load) {
            min_load = machine_loads[i];
            min_index = static_cast<int>(i);
        }
    }
    return min_index;
}

/**
 * @brief Algorytm List Scheduling (LSA).
 */
long long list_scheduling(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    if (num_machines <= 0) return 0;
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    for (int duration : task_times) {
        int target_machine = find_least_loaded_machine(machine_loads);
        if (target_machine != -1) {
            machine_loads[target_machine] += duration;
        } else {
            return -1; // Błąd
        }
    }
    if (machine_loads.empty()) return 0; // Should not happen if m>0
    return *std::max_element(machine_loads.begin(), machine_loads.end());
}

/**
 * @brief Algorytm Longest Processing Time First (LPT).
 */
long long longest_processing_time(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    if (num_machines <= 0) return 0;
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    std::vector<int> sorted_tasks = task_times;
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    return list_scheduling(num_machines, sorted_tasks, machine_loads); // Re-use LSA logic
}

/**
 * @brief Algorytm Programowania Dynamicznego (DP) dla P2||Cmax.
 */
long long dynamic_programming_p2(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm DP jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    long long total_sum = 0;
    for (int duration : task_times) {
        if (duration < 0) throw std::invalid_argument("Czas zadania nie moze byc ujemny.");
        if (total_sum > std::numeric_limits<long long>::max() - duration) throw std::overflow_error("Przepełnienie podczas sumowania czasow zadan dla DP.");
        total_sum += duration;
    }

    long long target_sum = total_sum / 2;
    const long long DP_SUM_LIMIT = 100000000;
    if (target_sum > DP_SUM_LIMIT) throw std::runtime_error("Suma docelowa (S/2 = " + std::to_string(target_sum) + ") jest zbyt duza dla DP.");

    std::vector<bool> dp;
    try {
        dp.resize(static_cast<size_t>(target_sum) + 1, false);
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("Nie mozna zaalokowac pamieci dla DP (rozmiar: " + std::to_string(target_sum + 1) + ")");
    }
    dp[0] = true;

    long long max_reachable_sum = 0;
    for (int duration : task_times) {
        if (duration <= 0) continue;
        long long current_task_time = static_cast<long long>(duration);
        for (long long k = std::min(max_reachable_sum + current_task_time, target_sum); k >= current_task_time; --k) {
            if (dp[static_cast<size_t>(k - current_task_time)]) {
                dp[static_cast<size_t>(k)] = true;
                max_reachable_sum = std::max(max_reachable_sum, k);
            }
        }
        if (target_sum >= 0 && static_cast<size_t>(target_sum) < dp.size() && dp[static_cast<size_t>(target_sum)]) { // Check bounds before access
            max_reachable_sum = target_sum;
        }
    }

    long long best_sum_machine1 = 0;
    for (long long k = target_sum; k >= 0; --k) {
        if (static_cast<size_t>(k) < dp.size() && dp[static_cast<size_t>(k)]) { // Check bounds before access
            best_sum_machine1 = k;
            break;
        }
    }

    long long load_machine1 = best_sum_machine1;
    long long load_machine2 = total_sum - best_sum_machine1;
    machine_loads[0] = load_machine1;
    machine_loads[1] = load_machine2;
    return std::max(load_machine1, load_machine2);
}

/**
 * @brief Algorytm Przeglądu Zupełnego (Brute Force) dla P2||Cmax.
 */
long long brute_force_p2(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm Brute Force jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size();
    const int MAX_N_FOR_BRUTE_FORCE = 25;
    if (n > MAX_N_FOR_BRUTE_FORCE) throw std::runtime_error("Liczba zadan (n=" + std::to_string(n) + ") jest zbyt duza dla Brute Force (limit: " + std::to_string(MAX_N_FOR_BRUTE_FORCE) + ").");
    if (n >= 64) throw std::overflow_error("Liczba zadan (n=" + std::to_string(n) + ") zbyt duza dla operacji bitowych (>= 64).");

    for (int duration : task_times) if (duration < 0) throw std::invalid_argument("Czas zadania nie moze byc ujemny.");

    long long min_found_cmax = std::numeric_limits<long long>::max();
    long long optimal_load1 = -1, optimal_load2 = -1;
    long long num_assignments = 1LL << n;

    for (long long i = 0; i < num_assignments; ++i) {
        long long current_load1 = 0, current_load2 = 0;
        for (int j = 0; j < n; ++j) {
            if (((i >> j) & 1) == 1) current_load2 += task_times[j];
            else current_load1 += task_times[j];
        }
        long long current_cmax = std::max(current_load1, current_load2);
        if (current_cmax < min_found_cmax) {
            min_found_cmax = current_cmax;
            optimal_load1 = current_load1;
            optimal_load2 = current_load2;
        }
    }

    if (optimal_load1 != -1) {
        machine_loads[0] = optimal_load1;
        machine_loads[1] = optimal_load2;
    } else if (n > 0) return -1; // Error

    return min_found_cmax;
}

/**
 * @brief Algorytm PTAS (Polynomial Time Approximation Scheme) dla P2||Cmax.
 */
long long ptas_p2(int num_machines, const std::vector<int>& task_times, int ptas_k, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm PTAS jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size();
    if (ptas_k <= 0) return longest_processing_time(num_machines, task_times, machine_loads); // Fallback to LPT

    int k_actual = std::min(ptas_k, n);
    std::vector<int> sorted_tasks = task_times;
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    long long min_partial_cmax = std::numeric_limits<long long>::max();
    long long best_partial_load1 = 0, best_partial_load2 = 0;
    const int MAX_K_FOR_PTAS_BF = 25;
    if (k_actual > MAX_K_FOR_PTAS_BF) throw std::runtime_error("Parametr k dla PTAS (k=" + std::to_string(k_actual) + ") jest zbyt duzy (limit: " + std::to_string(MAX_K_FOR_PTAS_BF) + ").");
    if (k_actual >= 64) throw std::overflow_error("Parametr k dla PTAS (k=" + std::to_string(k_actual) + ") zbyt duzy dla operacji bitowych (>= 64).");

    long long num_partial_assignments = 1LL << k_actual;
    for (long long i = 0; i < num_partial_assignments; ++i) {
        long long current_load1 = 0, current_load2 = 0;
        for (int j = 0; j < k_actual; ++j) {
            if (((i >> j) & 1) == 1) current_load2 += sorted_tasks[j];
            else current_load1 += sorted_tasks[j];
        }
        long long current_partial_cmax = std::max(current_load1, current_load2);
        if (current_partial_cmax < min_partial_cmax) {
            min_partial_cmax = current_partial_cmax;
            best_partial_load1 = current_load1;
            best_partial_load2 = current_load2;
        }
    }

    machine_loads[0] = best_partial_load1;
    machine_loads[1] = best_partial_load2;
    for (int j = k_actual; j < n; ++j) {
        if (machine_loads[0] <= machine_loads[1]) machine_loads[0] += sorted_tasks[j];
        else machine_loads[1] += sorted_tasks[j];
    }

    return std::max(machine_loads[0], machine_loads[1]);
}

/**
 * @brief Wewnętrzna funkcja DP dla FPTAS z możliwością backtrackingu.
 */
long long dp_for_fptas_with_backtracking(const std::vector<int>& scaled_tasks, long long target_sum_scaled, std::vector<int>& assignment_m1_indices) {
    int n = scaled_tasks.size();
    assignment_m1_indices.clear();
    if (target_sum_scaled < 0) return 0;

    std::vector<std::vector<bool>> dp_table;
    try {
        dp_table.resize(n + 1, std::vector<bool>(static_cast<size_t>(target_sum_scaled) + 1, false));
    } catch (const std::bad_alloc&) {
        throw std::runtime_error("FPTAS DP: Nie mozna zaalokowac pamieci (rozmiar: " + std::to_string(n + 1) + "x" + std::to_string(target_sum_scaled + 1) + ")");
    }
    dp_table[0][0] = true;

    for (int i = 1; i <= n; ++i) {
        long long current_scaled_task_time_ll = static_cast<long long>(scaled_tasks[i - 1]);
        for (long long k = 0; k <= target_sum_scaled; ++k) {
            dp_table[i][k] = dp_table[i - 1][k];
            if (current_scaled_task_time_ll <= k && !dp_table[i][k]) {
                if (k - current_scaled_task_time_ll >= 0 && dp_table[i - 1][static_cast<size_t>(k - current_scaled_task_time_ll)]) {
                    dp_table[i][k] = true;
                }
            }
        }
    }

    long long best_sum_scaled_m1 = 0;
    for (long long k = target_sum_scaled; k >= 0; --k) {
        if (dp_table[n][k]) { best_sum_scaled_m1 = k; break; }
    }

    long long current_sum = best_sum_scaled_m1;
    for (int i = n; i > 0 && current_sum > 0; --i) {
        // Bounds check before accessing dp_table
        if (current_sum < 0 || static_cast<size_t>(current_sum) >= dp_table[i-1].size()) {
            throw std::runtime_error("FPTAS Backtracking: Invalid current_sum index.");
        }

        if (!dp_table[i - 1][current_sum]) {
            int task_index = i - 1;
            assignment_m1_indices.push_back(task_index);
            if (task_index < 0 || static_cast<size_t>(task_index) >= scaled_tasks.size()) {
                throw std::runtime_error("FPTAS Backtracking: Invalid task_index.");
            }
            long long task_time = static_cast<long long>(scaled_tasks[task_index]);
            current_sum -= task_time;
        }
    }
    std::reverse(assignment_m1_indices.begin(), assignment_m1_indices.end());
    return best_sum_scaled_m1;
}


/**
 * @brief Algorytm FPTAS (Fully Polynomial Time Approximation Scheme) dla P2||Cmax.
 */
long long fptas_p2(int num_machines, const std::vector<int>& task_times, double epsilon, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm FPTAS jest zaimplementowany tylko dla m=2.");
    if (epsilon <= 0.0) throw std::invalid_argument("Parametr epsilon dla FPTAS musi byc wiekszy od 0.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size();
    long long total_sum_original = 0;
    for (int duration : task_times) {
        if (duration < 0) throw std::invalid_argument("FPTAS: Czas zadania nie moze byc ujemny.");
        if (total_sum_original > std::numeric_limits<long long>::max() - duration) throw std::overflow_error("FPTAS: Przepełnienie podczas sumowania oryginalnych czasow.");
        total_sum_original += duration;
    }
    if (total_sum_original == 0) return 0;

    double scale_factor_calc = std::floor(epsilon * static_cast<double>(total_sum_original) / (2.0 * n));
    long long fptas_k_scale = std::max(1LL, static_cast<long long>(scale_factor_calc));

    std::vector<int> scaled_task_times(n);
    long long total_sum_scaled = 0;
    for (int i = 0; i < n; ++i) {
        scaled_task_times[i] = static_cast<int>(task_times[i] / fptas_k_scale);
        total_sum_scaled += scaled_task_times[i];
    }

    long long target_sum_scaled = total_sum_scaled / 2;
    std::vector<int> machine1_task_indices;
    const long long FPTAS_DP_SUM_LIMIT = 200000000;
    if (target_sum_scaled > FPTAS_DP_SUM_LIMIT) throw std::runtime_error("FPTAS: Przeskalowana suma docelowa (S'/2 = " + std::to_string(target_sum_scaled) + ") jest zbyt duza dla DP.");

    dp_for_fptas_with_backtracking(scaled_task_times, target_sum_scaled, machine1_task_indices);

    long long final_load_m1 = 0, final_load_m2 = 0;
    std::vector<bool> assigned_to_m1(n, false);
    for (int index : machine1_task_indices) {
        if (index >= 0 && index < n) {
            final_load_m1 += task_times[index];
            assigned_to_m1[index] = true;
        } else throw std::runtime_error("FPTAS: Niepoprawny indeks zadania z backtrackingu DP.");
    }
    for (int i = 0; i < n; ++i) {
        if (!assigned_to_m1[i]) final_load_m2 += task_times[i];
    }

    machine_loads[0] = final_load_m1;
    machine_loads[1] = final_load_m2;
    return std::max(final_load_m1, final_load_m2);
}

/**
 * @brief Generuje wektor zadań z losowymi czasami wykonania.
 */
std::vector<int> generate_tasks(int num_tasks, int min_p, int max_p) {
    if (num_tasks < 0 || min_p < 0 || max_p < min_p) throw std::invalid_argument("Nieprawidlowe argumenty dla generate_tasks.");
    static std::random_device randomDevice;
    static std::mt19937 randomNumberEngine(randomDevice());
    std::uniform_int_distribution<int> distribution(min_p, max_p);
    std::vector<int> tasks(num_tasks);
    for (int i = 0; i < num_tasks; ++i) tasks[i] = distribution(randomNumberEngine);
    return tasks;
}

// Typ wskaźnika na funkcję algorytmu
using SchedulingAlgorithmFunc = std::function<long long(int, const std::vector<int>&, std::vector<long long>&)>;

/**
 * @brief Mierzy czas wykonania i zbiera wynik podanego algorytmu.
 */
AlgorithmRunResult measure_algorithm_run(SchedulingAlgorithmFunc algorithm_func, int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    AlgorithmRunResult result;
    std::vector<long long> current_loads;
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        result.cmax = algorithm_func(num_machines, task_times, current_loads);
        auto end_time = std::chrono::high_resolution_clock::now();
        if (result.cmax >= 0) {
            auto duration_object = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            result.duration_us = duration_object.count();
            result.success = true;
        } else {
            result.error_message = "Algorytm zwrocil blad (Cmax < 0)";
            result.success = false;
        }
    } catch (const std::exception& e) {
        result.error_message = "Wyjatek: " + std::string(e.what());
        result.success = false;
    } catch (...) {
        result.error_message = "Nieznany wyjatek";
        result.success = false;
    }
    return result;
}

/**
 * @brief Formatuje wartość liczbową lub status błędu do stringa dla komórki tabeli.
 */
std::string format_cell(long long value, int precision, const std::string& status_if_error = "ERROR") {
    std::stringstream cell_stream;
    if (value >= 0) {
        if (precision == 1) cell_stream << std::fixed << std::setprecision(1) << static_cast<double>(value); // Dla Cmax
        else cell_stream << value; // Dla czasu (całkowite us)
    } else {
        cell_stream << status_if_error;
    }
    return cell_stream.str();
}

// --- Główna Funkcja Programu ---
int main() {
    // --- Konfiguracja Eksperymentu ---
    const int NUM_MACHINES = 2;
    const int PTAS_K_PARAM = 5;
    const double FPTAS_EPSILON_PARAM = 0.1;

    // Lista konfiguracji do przetestowania
    std::vector<InstanceConfig> test_configurations = {
            {10, 1, 10}, {10, 10, 20},
            {15, 1, 10},
            {20, 1, 10}, {20, 10, 20},
            //{25, 1, 5},   // Może być bardzo wolne dla BF/PTAS
            //{50, 50, 100}, // Może powodować błędy pamięci DP/FPTAS
    };

    // --- Ustawienia Wyglądu Tabeli Wyników ---
    const int COL_WIDTH_N = 7;
    const int COL_WIDTH_RANGE = 12;
    const int COL_WIDTH_CMAX = 14;
    const int COL_WIDTH_TIME = 17;
    const int TOTAL_TABLE_WIDTH = COL_WIDTH_N + COL_WIDTH_RANGE + 6 * (COL_WIDTH_CMAX + COL_WIDTH_TIME) + 15;

    // --- Drukowanie Nagłówka Tabeli ---
    std::cout << "\n--- Tabela Wynikow Eksperymentow (P2||Cmax) ---\n";
    std::cout << "Jeden przebieg na konfiguracje.\n";
    std::cout << "Parametry aproksymacyjne: PTAS k=" << PTAS_K_PARAM << ", FPTAS epsilon=" << FPTAS_EPSILON_PARAM << "\n\n";
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl;
    std::cout << std::left << "|"
              << std::setw(COL_WIDTH_N) << " n" << "|"
              << std::setw(COL_WIDTH_RANGE) << " p_j range" << "|"
              << std::setw(COL_WIDTH_CMAX) << " LSA Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " LSA Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " LPT Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " LPT Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " DP Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " DP Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " BF Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " BF Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " PTAS Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " PTAS Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " FPTAS Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " FPTAS Time(us)" << "|"
              << std::endl;
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl;

    // --- Pętla Główna - Testowanie Konfiguracji ---
    for (const InstanceConfig& config : test_configurations) {

        std::vector<int> current_tasks;
        try {
            current_tasks = generate_tasks(config.num_tasks, config.min_task_time, config.max_task_time);
        } catch (const std::exception& e) {
            std::cerr << "KRYTYCZNY BLAD GENEROWANIA ZADAN dla " << config.description() << ": " << e.what() << std::endl;
            std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                      << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";
            std::cout << std::right;
            for(int i=0; i<6; ++i) { std::cout << std::setw(COL_WIDTH_CMAX) << "CRITICAL" << "|" << std::setw(COL_WIDTH_TIME) << "CRITICAL" << "|"; }
            std::cout << std::endl;
            continue;
        }

        std::vector<long long> temp_loads; // Pomocniczy wektor

        // Uruchom i zmierz wszystkie algorytmy
        AlgorithmRunResult res_lsa = measure_algorithm_run(list_scheduling, NUM_MACHINES, current_tasks, temp_loads);
        AlgorithmRunResult res_lpt = measure_algorithm_run(longest_processing_time, NUM_MACHINES, current_tasks, temp_loads);
        AlgorithmRunResult res_dp = measure_algorithm_run(dynamic_programming_p2, NUM_MACHINES, current_tasks, temp_loads);
        AlgorithmRunResult res_bf = measure_algorithm_run(brute_force_p2, NUM_MACHINES, current_tasks, temp_loads);
        AlgorithmRunResult res_ptas = measure_algorithm_run(
                [&](int m, const auto& t, auto& l){ return ptas_p2(m, t, PTAS_K_PARAM, l); },
                NUM_MACHINES, current_tasks, temp_loads);
        AlgorithmRunResult res_fptas = measure_algorithm_run(
                [&](int m, const auto& t, auto& l){ return fptas_p2(m, t, FPTAS_EPSILON_PARAM, l); },
                NUM_MACHINES, current_tasks, temp_loads);

        // --- Drukowanie Wiersza Wyników dla Konfiguracji ---
        std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                  << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";
        std::cout << std::right; // Ustawienie wyrównania dla reszty kolumn

        // Funkcja pomocnicza do drukowania pary komórek dla wyniku
        auto print_result_pair = [&](const AlgorithmRunResult& result) {
            std::string cmax_str, time_str, status = "N/A";
            if (!result.success) {
                if (result.error_message.find("zbyt duz") != std::string::npos) status = "TOO LARGE";
                else if (result.error_message.find("pamieci") != std::string::npos) status = "MEM/LIMIT";
                else status = "ERROR";
            }
            if (result.success) {
                cmax_str = format_cell(result.cmax, 1, ""); // 1 miejsce dla Cmax
                time_str = format_cell(result.duration_us, 0, ""); // 0 miejsc dla czasu
            } else {
                cmax_str = status; time_str = status;
            }
            std::cout << std::setw(COL_WIDTH_CMAX) << cmax_str << "|"
                      << std::setw(COL_WIDTH_TIME) << time_str << "|";
        };

        // Wydrukuj wyniki dla wszystkich algorytmów
        print_result_pair(res_lsa); print_result_pair(res_lpt); print_result_pair(res_dp);
        print_result_pair(res_bf); print_result_pair(res_ptas); print_result_pair(res_fptas);
        std::cout << std::endl; // Koniec wiersza tabeli

        // Funkcja pomocnicza do drukowania szczegółów błędu (jeśli status to ERROR)
        auto print_error_info_if_needed = [&](const std::string& algo_name, const AlgorithmRunResult& result) {
            std::string status = result.success ? "" : (result.error_message.find("zbyt duz") != std::string::npos ? "TOO LARGE" : (result.error_message.find("pamieci") != std::string::npos ? "MEM/LIMIT" : "ERROR"));
            if (status == "ERROR") {
                std::cout << std::string(1, '|') << std::string(COL_WIDTH_N + COL_WIDTH_RANGE + 2, ' ')
                          << "[" << algo_name << " Error Info: " << result.error_message << "]" << std::endl;
            }
        };

        // Wydrukuj szczegóły błędów, jeśli są typu "ERROR"
        print_error_info_if_needed("DP", res_dp); print_error_info_if_needed("BF", res_bf);
        print_error_info_if_needed("PTAS", res_ptas); print_error_info_if_needed("FPTAS", res_fptas);

    } // Koniec pętli po konfiguracjach (config)

    // --- Drukowanie Stopki Tabeli ---
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl; // Linia dolna
    std::cout << "Koniec eksperymentow.\n" << std::endl;

    return 0; // Zakończ program pomyślnie
}