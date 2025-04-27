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
#include <utility>  // Dla std::pair (chociaż nie jest już bezpośrednio używane po refaktoryzacji)
#include <cmath>    // Dla std::floor (chociaż dzielenie całkowite też działa)
#include <stdexcept>// Do obsługi standardowych wyjątków (np. invalid_argument)
#include <sstream>  // Dla std::stringstream (konwersja liczb na string, formatowanie)

// Definicja typu dla czasów zadań i obciążeń maszyn (używamy long long dla dużych sum)
using TimeType = long long;

// --- Funkcje pomocnicze i algorytmy ---

/**
 * @brief Znajduje indeks maszyny z najmniejszym aktualnym obciążeniem.
 * @param machine_loads Wektor z aktualnymi obciążeniami maszyn.
 * @return Indeks maszyny z najmniejszym obciążeniem lub -1 jeśli wektor jest pusty.
 */
int find_least_loaded_machine(const std::vector<TimeType>& machine_loads) {
    if (machine_loads.empty()) {
        return -1; // Błąd: brak maszyn
    }
    int min_index = 0; // Indeks maszyny z minimalnym obciążeniem
    TimeType min_load = machine_loads[0]; // Minimalne znalezione obciążenie

    // Pętla po pozostałych maszynach
    for (size_t i = 1; i < machine_loads.size(); ++i) {
        if (machine_loads[i] < min_load) {
            min_load = machine_loads[i];
            min_index = static_cast<int>(i); // Rzutowanie size_t na int
        }
    }
    return min_index;
}

/**
 * @brief Algorytm List Scheduling (LSA). Przydziela zadania w podanej kolejności.
 * @param num_machines Liczba maszyn (m).
 * @param task_times Wektor czasów wykonania zadań.
 * @param machine_loads (Wyjście) Wektor, który zostanie wypełniony końcowymi obciążeniami maszyn.
 * @return Obliczony Cmax (maksymalne obciążenie) lub 0 jeśli brak zadań/maszyn, -1 w razie błędu.
 */
TimeType list_scheduling(int num_machines, const std::vector<int>& task_times, std::vector<TimeType>& machine_loads) {
    if (num_machines <= 0) {
        return 0; // Brak maszyn
    }
    // Inicjalizacja wektora obciążeń zerami
    machine_loads.assign(num_machines, 0LL); // 0LL oznacza zero typu long long
    if (task_times.empty()) {
        return 0; // Brak zadań
    }

    // Przydzielanie zadań
    for (int duration : task_times) {
        int target_machine = find_least_loaded_machine(machine_loads);
        if (target_machine != -1) {
            machine_loads[target_machine] += duration;
        } else {
            std::cerr << "Blad w LSA: Nie znaleziono maszyny docelowej.\n";
            return -1; // Wewnętrzny błąd (nie powinno się zdarzyć przy m > 0)
        }
    }

    // Znalezienie maksymalnego obciążenia (Cmax)
    return *std::max_element(machine_loads.begin(), machine_loads.end());
}

/**
 * @brief Algorytm Longest Processing Time First (LPT). Sortuje zadania malejąco i używa LSA.
 * @param num_machines Liczba maszyn (m).
 * @param task_times Wektor czasów wykonania zadań.
 * @param machine_loads (Wyjście) Wektor, który zostanie wypełniony końcowymi obciążeniami maszyn.
 * @return Obliczony Cmax lub 0 jeśli brak zadań/maszyn, -1 w razie błędu.
 */
TimeType longest_processing_time(int num_machines, const std::vector<int>& task_times, std::vector<TimeType>& machine_loads) {
    if (num_machines <= 0) {
        return 0;
    }
    if (task_times.empty()) {
        machine_loads.assign(num_machines, 0LL); // Ustaw obciążenia na 0 nawet przy braku zadań
        return 0;
    }

    // Stworzenie kopii wektora zadań do posortowania
    std::vector<int> sorted_tasks = task_times;
    // Sortowanie zadań malejąco według czasu wykonania
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    // Wywołanie logiki LSA na posortowanych zadaniach
    // (ponowne wywołanie list_scheduling jest bardziej przejrzyste niż kopiowanie kodu)
    return list_scheduling(num_machines, sorted_tasks, machine_loads);
}

/**
 * @brief Algorytm Programowania Dynamicznego (DP) dla P2||Cmax (tylko dla 2 maszyn).
 * @param num_machines Liczba maszyn (musi być 2).
 * @param task_times Wektor czasów wykonania zadań.
 * @param machine_loads (Wyjście) Wektor, który zostanie wypełniony optymalnymi obciążeniami maszyn.
 * @return Optymalny Cmax lub 0 jeśli brak zadań. Rzuca wyjątek dla m != 2 lub innych błędów.
 */
TimeType dynamic_programming_p2(int num_machines, const std::vector<int>& task_times, std::vector<TimeType>& machine_loads) {
    if (num_machines != 2) {
        throw std::invalid_argument("Algorytm DP jest zaimplementowany tylko dla m=2.");
    }
    machine_loads.assign(num_machines, 0LL); // Zawsze 2 maszyny
    if (task_times.empty()) {
        return 0;
    }

    // 1. Oblicz sumę wszystkich czasów zadań (S)
    TimeType total_sum = 0;
    for (int duration : task_times) {
        if (duration < 0) {
            throw std::invalid_argument("Czas zadania nie moze byc ujemny.");
        }
        // Sprawdzenie przepełnienia przed dodaniem
        if (total_sum > std::numeric_limits<TimeType>::max() - duration) {
            throw std::overflow_error("Przepełnienie podczas sumowania czasow zadan dla DP.");
        }
        total_sum += duration;
    }

    // 2. Oblicz docelową sumę dla jednej maszyny (K = S / 2)
    TimeType target_sum = total_sum / 2;

    // Sprawdzenie limitu rozmiaru tablicy DP, aby uniknąć problemów z pamięcią
    const TimeType DP_SUM_LIMIT = 100000000; // Limit np. 100 milionów
    if (target_sum > DP_SUM_LIMIT) {
        throw std::runtime_error("Suma docelowa (S/2 = " + std::to_string(target_sum) + ") jest zbyt duza dla DP.");
    }

    // 3. Stwórz i zainicjalizuj tablicę DP
    // dp[k] będzie 'true', jeśli suma 'k' jest osiągalna
    std::vector<bool> dp;
    try {
        // Rozmiar target_sum + 1, aby mieć indeksy od 0 do target_sum
        dp.resize(static_cast<size_t>(target_sum) + 1, false);
    } catch (const std::bad_alloc&) { // Złap konkretny błąd braku pamięci
        throw std::runtime_error("Nie mozna zaalokowac pamieci dla DP (rozmiar: "
                                 + std::to_string(target_sum + 1) + ")");
    }
    dp[0] = true; // Suma 0 jest zawsze osiągalna (nie biorąc żadnego zadania)

    // 4. Wypełnij tablicę DP
    TimeType max_reachable_sum = 0; // Śledzi największą dotąd osiągniętą sumę <= target_sum
    for (int duration : task_times) {
        if (duration <= 0) continue; // Pomiń zadania o zerowym/ujemnym czasie
        TimeType current_task_time = static_cast<TimeType>(duration);

        // Pętla idzie od tyłu, aby każde zadanie zostało użyte co najwyżej raz w danym kroku
        for (TimeType k = std::min(max_reachable_sum + current_task_time, target_sum); k >= current_task_time; --k) {
            // Jeśli suma (k - czas_zadania) była osiągalna...
            if (dp[static_cast<size_t>(k - current_task_time)]) {
                // ...to suma k też staje się osiągalna
                dp[static_cast<size_t>(k)] = true;
                // Zaktualizuj maksymalną osiągalną sumę
                max_reachable_sum = std::max(max_reachable_sum, k);
            }
        }
        // Jeśli osiągnęliśmy dokładnie target_sum, możemy zaktualizować max_reachable_sum
        if (dp[static_cast<size_t>(target_sum)]) {
            max_reachable_sum = target_sum;
            // Teoretycznie można by przerwać zewnętrzną pętlę, ale nie ma to dużego wpływu
        }
    }

    // 5. Znajdź najlepszą (największą) sumę osiągalną dla maszyny 1, która jest <= target_sum
    TimeType best_sum_machine1 = 0;
    for (TimeType k = target_sum; k >= 0; --k) {
        if (dp[static_cast<size_t>(k)]) {
            best_sum_machine1 = k;
            break; // Znaleźliśmy największą, możemy przerwać
        }
    }

    // 6. Oblicz obciążenia obu maszyn i Cmax
    TimeType load_machine1 = best_sum_machine1;
    TimeType load_machine2 = total_sum - best_sum_machine1;
    TimeType cmax = std::max(load_machine1, load_machine2);

    // Zapisz wynikowe obciążenia
    machine_loads[0] = load_machine1;
    machine_loads[1] = load_machine2;

    return cmax;
}

/**
 * @brief Generuje wektor zadań z losowymi czasami wykonania.
 * @param num_tasks Liczba zadań do wygenerowania (n).
 * @param min_p Minimalny czas zadania.
 * @param max_p Maksymalny czas zadania.
 * @return Wektor z czasami zadań. Rzuca wyjątek przy błędnych parametrach.
 */
std::vector<int> generate_tasks(int num_tasks, int min_p, int max_p) {
    if (num_tasks < 0 || min_p < 0 || max_p < min_p) {
        throw std::invalid_argument("Nieprawidlowe argumenty dla generate_tasks.");
    }

    // Inicjalizacja generatora liczb losowych (tylko raz przy pierwszym wywołaniu)
    static std::random_device randomDevice; // Źródło prawdziwej losowości (ziarno)
    static std::mt19937 randomNumberEngine(randomDevice()); // Generator Mersenne Twister

    // Rozkład jednostajny dla liczb całkowitych w zakresie [min_p, max_p]
    std::uniform_int_distribution<int> distribution(min_p, max_p);

    std::vector<int> tasks(num_tasks);
    for (int i = 0; i < num_tasks; ++i) {
        tasks[i] = distribution(randomNumberEngine); // Generuj i zapisz losowy czas
    }
    return tasks;
}

// Struktura przechowująca wynik pojedynczego uruchomienia algorytmu
struct AlgorithmRunResult {
    TimeType cmax = -1;          // Wynik Cmax (-1 oznacza błąd)
    long long duration_us = -1;  // Czas wykonania w mikrosekundach (-1 oznacza błąd)
    bool success = false;        // Czy wykonanie się powiodło?
    std::string error_message = ""; // Komunikat błędu, jeśli wystąpił
};

// Typ wskaźnika na funkcję algorytmu szeregowania
// Akceptuje funkcje o sygnaturze: TimeType(int, const std::vector<int>&, std::vector<TimeType>&)
using SchedulingAlgorithmFunc = std::function<TimeType(int, const std::vector<int>&, std::vector<TimeType>&)>;

/**
 * @brief Mierzy czas wykonania i zbiera wynik podanego algorytmu szeregowania.
 * @param algorithm_func Funkcja algorytmu do uruchomienia.
 * @param num_machines Liczba maszyn.
 * @param task_times Wektor czasów zadań.
 * @param machine_loads Pomocniczy wektor na obciążenia (nie jest używany na zewnątrz).
 * @return Struktura AlgorithmRunResult z wynikiem, czasem i statusem.
 */
AlgorithmRunResult measure_algorithm_run(
        SchedulingAlgorithmFunc algorithm_func,
        int num_machines,
        const std::vector<int>& task_times,
        std::vector<TimeType>& machine_loads // Tylko jako placeholder dla sygnatury funkcji
) {
    AlgorithmRunResult result;
    std::vector<TimeType> current_loads; // Lokalna kopia obciążeń dla tego uruchomienia

    try {
        // Pomiar czasu startu
        auto start_time = std::chrono::high_resolution_clock::now();

        // Wywołanie algorytmu
        result.cmax = algorithm_func(num_machines, task_times, current_loads);

        // Pomiar czasu końca
        auto end_time = std::chrono::high_resolution_clock::now();

        // Sprawdzenie, czy algorytm nie zgłosił błędu wewnętrznie (np. zwracając -1)
        if (result.cmax >= 0) {
            // Obliczenie czasu trwania
            auto duration_object = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            result.duration_us = duration_object.count();
            result.success = true;
        } else {
            result.error_message = "Algorytm zwrocil wewnetrzny blad (Cmax < 0)";
            result.success = false;
        }

    } catch (const std::exception& e) {
        // Złapanie standardowych wyjątków (np. brak pamięci, zły argument)
        result.error_message = "Wyjatek: " + std::string(e.what());
        result.success = false;
    } catch (...) {
        // Złapanie innych, nieznanych wyjątków
        result.error_message = "Nieznany wyjatek";
        result.success = false;
    }

    return result;
}


// --- Główna funkcja programu ---

int main() {
    // --- Konfiguracja Eksperymentu ---
    const int NUM_MACHINES = 2;              // Liczba maszyn (stała dla P2||Cmax)
    const int RUNS_PER_INSTANCE = 10;        // Ile razy powtórzyć test dla każdej konfiguracji

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

    // Lista konfiguracji do przetestowania
    std::vector<InstanceConfig> test_configurations = {
            {10, 1, 10}, {10, 10, 20}, {20, 1, 10}, {20, 10, 20}, {20, 50, 100},
            {50, 1, 10}, {50, 10, 20}, {50, 50, 100}, {30, 500, 1000},
            // Odkomentuj, aby przetestować przypadek z potencjalnym błędem DP (duże S)
            // {200, 1000000, 2000000}
    };

    // --- Ustawienia Wyglądu Tabeli Wyników ---
    const int COL_WIDTH_N = 7;      // Szerokość kolumny 'n'
    const int COL_WIDTH_RANGE = 12; // Szerokość kolumny 'p_j range'
    const int COL_WIDTH_CMAX = 14;  // Szerokość kolumny 'Avg Cmax'
    const int COL_WIDTH_TIME = 17;  // Szerokość kolumny 'Avg Time(us)'
    // Całkowita szerokość tabeli (suma szerokości + 9 separatorów)
    const int TOTAL_TABLE_WIDTH = COL_WIDTH_N + COL_WIDTH_RANGE
                                  + 3 * (COL_WIDTH_CMAX + COL_WIDTH_TIME) + 9;

    // --- Drukowanie Nagłówka Tabeli ---
    std::cout << "\n--- Tabela Wynikow Eksperymentow (P2||Cmax) ---\n";
    std::cout << "Liczba powtorzen na konfiguracje: " << RUNS_PER_INSTANCE << "\n\n";
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl; // Linia górna

    // Wiersz z nazwami kolumn (wyrównane do lewej)
    std::cout << std::left << "|"
              << std::setw(COL_WIDTH_N) << " n" << "|"
              << std::setw(COL_WIDTH_RANGE) << " p_j range" << "|"
              << std::setw(COL_WIDTH_CMAX) << " Avg LSA Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " Avg LSA Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " Avg LPT Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " Avg LPT Time(us)" << "|"
              << std::setw(COL_WIDTH_CMAX) << " Avg DP Cmax" << "|"
              << std::setw(COL_WIDTH_TIME) << " Avg DP Time(us)" << "|"
              << std::endl;
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl; // Linia pod nagłówkiem

    // --- Pętla Główna - Testowanie Konfiguracji ---
    for (const InstanceConfig& config : test_configurations) {

        // Zmienne do sumowania wyników dla danej konfiguracji
        double sum_cmax_lsa = 0, sum_cmax_lpt = 0, sum_cmax_dp = 0;
        double sum_duration_lsa = 0, sum_duration_lpt = 0, sum_duration_dp = 0;
        int successful_runs_lsa = 0, successful_runs_lpt = 0, successful_runs_dp = 0;
        std::string first_dp_error_msg = ""; // Zapisujemy pierwszy błąd DP

        // Pętla powtórzeń dla uśrednienia
        for (int run = 0; run < RUNS_PER_INSTANCE; ++run) {
            std::vector<int> current_tasks;
            try {
                // Wygeneruj nową instancję zadań
                current_tasks = generate_tasks(config.num_tasks, config.min_task_time, config.max_task_time);
            } catch (const std::exception& e) {
                std::cerr << "KRYTYCZNY BLAD: Nie udalo sie wygenerowac zadan dla "
                          << config.description() << ": " << e.what() << std::endl;
                // Oznacz wszystkie algorytmy jako nieudane dla tej konfiguracji
                successful_runs_lsa = successful_runs_lpt = successful_runs_dp = -1; // Używamy -1 jako flagi błędu krytycznego
                break; // Przerwij pętlę powtórzeń dla tej konfiguracji
            }

            std::vector<TimeType> temp_machine_loads; // Pomocniczy wektor

            // Uruchom i zmierz LSA
            AlgorithmRunResult lsa_result = measure_algorithm_run(list_scheduling, NUM_MACHINES, current_tasks, temp_machine_loads);
            if (lsa_result.success) {
                sum_cmax_lsa += lsa_result.cmax;
                sum_duration_lsa += lsa_result.duration_us;
                successful_runs_lsa++;
            }

            // Uruchom i zmierz LPT
            AlgorithmRunResult lpt_result = measure_algorithm_run(longest_processing_time, NUM_MACHINES, current_tasks, temp_machine_loads);
            if (lpt_result.success) {
                sum_cmax_lpt += lpt_result.cmax;
                sum_duration_lpt += lpt_result.duration_us;
                successful_runs_lpt++;
            }

            // Uruchom i zmierz DP
            AlgorithmRunResult dp_result = measure_algorithm_run(dynamic_programming_p2, NUM_MACHINES, current_tasks, temp_machine_loads);
            if (dp_result.success) {
                sum_cmax_dp += dp_result.cmax;
                sum_duration_dp += dp_result.duration_us;
                successful_runs_dp++;
            } else {
                // Zapisz tylko pierwszy napotkany błąd DP
                if (first_dp_error_msg.empty()) {
                    first_dp_error_msg = dp_result.error_message;
                }
            }
        } // Koniec pętli powtórzeń (run)

        // --- Drukowanie Wiersza Wyników dla Konfiguracji ---

        // Funkcja pomocnicza do formatowania pojedynczej wartości w komórce tabeli
        auto format_cell = [&](double value, int precision, const std::string& status_if_not_ok = "N/A") -> std::string {
            std::stringstream cell_stream;
            if (value >= 0) { // Jeśli wartość jest poprawna (nie -1)
                cell_stream << std::fixed << std::setprecision(precision) << value;
            } else {
                cell_stream << status_if_not_ok;
            }
            return cell_stream.str();
        };

        // Drukowanie pierwszych dwóch kolumn (n i zakres) - wyrównane do lewej
        std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                  << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";

        // Drukowanie kolumn z wynikami algorytmów - wyrównane do prawej
        // Używamy std::right przed wypisaniem wartości numerycznych
        std::cout << std::right;

        // Funkcja pomocnicza do drukowania pary komórek (Cmax, Czas) dla jednego algorytmu
        auto print_result_pair = [&](int successful_runs, double total_cmax, double total_time, const std::string& error_status = "N/A") {
            std::string cmax_str, time_str;
            if (successful_runs > 0) {
                cmax_str = format_cell(total_cmax / successful_runs, 1); // 1 miejsce po przecinku dla Cmax
                time_str = format_cell(total_time / successful_runs, 2); // 2 miejsca dla czasu
            } else {
                cmax_str = (successful_runs == -1) ? "CRITICAL" : error_status; // Użyj statusu błędu lub N/A
                time_str = cmax_str; // Czas oznaczamy tak samo jak Cmax w razie błędu
            }
            std::cout << std::setw(COL_WIDTH_CMAX) << cmax_str << "|"
                      << std::setw(COL_WIDTH_TIME) << time_str << "|";
        };

        // Wydrukuj wyniki dla LSA, LPT, DP
        print_result_pair(successful_runs_lsa, sum_cmax_lsa, sum_duration_lsa);
        print_result_pair(successful_runs_lpt, sum_cmax_lpt, sum_duration_lpt);
        // Dla DP przekazujemy specjalny status "ERROR", jeśli były błędy
        std::string dp_status = (successful_runs_dp == 0 && !first_dp_error_msg.empty()) ? "ERROR" : "N/A";
        print_result_pair(successful_runs_dp, sum_cmax_dp, sum_duration_dp, dp_status);

        std::cout << std::endl; // Koniec wiersza tabeli

        // Jeśli wystąpił błąd DP (i nie był to błąd krytyczny), wydrukuj informację
        if (!first_dp_error_msg.empty() && successful_runs_dp >= 0 && successful_runs_dp < RUNS_PER_INSTANCE) {
            std::cout << std::string(1, '|') // Wyrównanie do lewej krawędzi tabeli
                      << std::string(COL_WIDTH_N + COL_WIDTH_RANGE + 2, ' ') // Wcięcie, aby zacząć pod kolumnami LSA
                      << "[DP Info: " << successful_runs_dp << "/" << RUNS_PER_INSTANCE << " OK. First error: " << first_dp_error_msg << "]"
                      << std::endl;
        }

    } // Koniec pętli po konfiguracjach (config)

    // --- Drukowanie Stopki Tabeli ---
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl; // Linia dolna
    std::cout << "Koniec eksperymentow.\n" << std::endl;

    return 0; // Zakończ program pomyślnie
}