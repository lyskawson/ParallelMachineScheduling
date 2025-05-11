#include <iostream>
#include <vector>
#include <numeric>  // std::max_element
#include <algorithm>// std::sort, std::min_element, std::max_element, std::min, std::max
#include <limits>   // std::numeric_limits
#include <functional> // std::greater
#include <random>
#include <chrono>
#include <iomanip>
#include <string>
#include <cmath>
#include <stdexcept>
#include <sstream>


//konfigruacja testowa
struct InstanceConfig {
    int num_tasks;      // Liczba zadan (n)
    int min_task_time;  // Minimalny czas zadania (min_p)
    int max_task_time;  // Maksymalny czas zadania (max_p)


    // Funkcja pomocnicza do zakresu czasow dla tabeli
    std::string range_string_formatted() const {
        std::stringstream ss;
        ss << "[" << std::setw(3) << std::right << min_task_time << "-"
           << std::setw(3) << std::right << max_task_time << "]";
        return ss.str();
    }
};

//zwraca indeks maszyny z najmniejszym obciazeniem
int find_least_loaded_machine(const std::vector<long long> &machine_loads) {
    // Sprawdzenie, czy sa jakies maszyny
    if (machine_loads.empty()) {
        return -1; // Blad
    }
    // Zakladamy, ze pierwsza maszyna jest na razie najmniej obciazona
    int min_index = 0;
    long long min_load = machine_loads[0];
    // Przechodzimy przez pozostale maszyny
    for (size_t i = 1; i < machine_loads.size(); ++i) {
        // Jesli znaleźlismy maszyne z mniejszym obciazeniem
        if (machine_loads[i] < min_load) {
            // Zapisujemy jej obciazenie i indeks
            min_load = machine_loads[i];
            min_index = static_cast<int>(i);
        }
    }
    // Zwracamy indeks maszyny ktora zwolni sie najwczesniej
    return min_index;
}


//algorytm LSA: Przechodzi przez zadania w podanej kolejnosci.Kazde zadanie przypisuje do maszyny, ktora aktualnie ma najmniejsze obciazenie
long long
list_scheduling_algorithm(int num_machines, const std::vector<int> &task_times, std::vector<long long> &machine_loads) {

    if (num_machines <= 0)
        return 0;

    // Wyzerowanie obciazen maszyn na poczatku
    machine_loads.assign(num_machines, 0LL); // Uzywa vector::assign do ustawienia rozmiaru i wartosci

    // Jesli nie ma zadan, Cmax wynosi 0
    if (task_times.empty())
        return 0;

    // petla przechodzi przez kazde zadanie
    for (int duration: task_times) {
        // Znajdź maszyne ktora skonczy najwczesniej
        int target_machine = find_least_loaded_machine(machine_loads);
        // Przypisz biezace zadanie do tej maszyny
        if (target_machine != -1) {
            machine_loads[target_machine] += duration;
        } else {
            return -1; //  blad
        }
    }

    // Po przypisaniu wszystkich zadan, Cmax to maksymalne obciazenie sposrod wszystkich maszyn
    if (machine_loads.empty())
        return 0; // Dodatkowe zabezpieczenie
    return *std::max_element(machine_loads.begin(), machine_loads.end()); // Uzywa std::max_element
}

//Algprytm LPT: Zanim zacznie przypisywać zadania, sortuje je malejaco wedlug czasow ich wykonania (najdluzsze najpierw)
//Nastepnie stosuje te sama logike co LSA: przypisuje kolejne (posortowane) zadanie do maszyny, ktora zwolni sie najwczesniej.

long long
longest_processing_time(int num_machines, const std::vector<int> &task_times, std::vector<long long> &machine_loads) {
    if (num_machines <= 0)
        return 0;
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty())
        return 0;

    //Utworz kopie wektora czasow zadan
    std::vector<int> sorted_tasks = task_times;
    //Posortuj kopie malejaco (najdluzsze zadania na poczatku)
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    // Zastosuj algorytm LSA na posortowanej liscie zadan
    return list_scheduling_algorithm(num_machines, sorted_tasks, machine_loads);
}


//Algorytm DP dla P2||Cmax: Znajduje rozwiazanie optymalne dla problemu podzialu zadan na DWIE maszyny. Problem jest rownowazny znalezieniu podzbioru zadan, ktorego suma czasow
// jest jak najblizsza (ale nie wieksza) polowie sumy wszystkich czasow (S/2).

long long
dynamic_programming_p2(int num_machines, const std::vector<int> &task_times, std::vector<long long> &machine_loads) {

    if (num_machines != 2) {
        throw std::invalid_argument("Algorytm DP jest zaimplementowany tylko dla m=2.");
    }

    // Inicjalizacja obciazen dla 2 maszyn
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty())
        return 0;

    // Oblicz calkowita sume czasow zadan (S)
    long long total_sum = 0;
    for (int duration: task_times) {
        total_sum += duration;
    }

    // Oblicz docelowa sume dla jednej maszyny (polowa sumy calkowitej)
    long long target_sum = total_sum / 2;

    // Stworz tablice dynamiczna 'dp'
    // dp[k] bedzie mialo wartosć 'true', jesli suma czasow 'k' jest mozliwa do osiagnieci przez jakis podzbior zadan
    // Rozmiar to target_sum + 1 (indeksy 0 do target_sum).
    std::vector<bool> dp(static_cast<size_t>(target_sum) + 1, false);
    // Inicjalizacja: suma 0 jest zawsze osiagalna
    dp[0] = true;

    // Wypelnianie tablicy DP
    long long max_reachable_sum_so_far = 0; // Pomocnicza zmienna
    // Petla po wszystkich zadaniach
    for (int duration: task_times) {
        if (duration <= 0)
            continue;
        long long current_task_time = static_cast<long long>(duration);

        // Petla wewnetrzna idzie od najwyzszej mozliwej sumy do czasu biezacego zadania min(max_reachable_sum_so_far + current_task_time, target_sum)
        //bo nie ma sensu sprawdzać sum wiekszych niz target_sum, ani tych ktore na pewno nie moga powstać przez dodanie current_task_time.

        for (long long k = std::min(max_reachable_sum_so_far + current_task_time, target_sum);
             k >= current_task_time; --k) {
            // Jesli suma (k - czas biezacego zadania) byla osiagalna wczesniej (sprawdzamy dp[k - current_task_time])
            if (dp[static_cast<size_t>(k - current_task_time)]) {
                // suma 'k' staje sie teraz osiagalna.
                dp[static_cast<size_t>(k)] = true;
                // Aktualizujemy najwieksza dotad znaleziona osiagalna sume
                max_reachable_sum_so_far = std::max(max_reachable_sum_so_far, k);
            }
        }
        // sprawdzenie czy nie osiagnelismy juz celu
        if (dp[static_cast<size_t>(target_sum)]) {
            max_reachable_sum_so_far = target_sum;
        }
    }

    // Znajdź najlepsza (najwieksza) sume dla maszyny 1, Szukamy od target_sum w dol, az znajdziemy pierwsza osiagalna sume
    long long best_sum_machine1 = 0;
    for (long long k = target_sum; k >= 0; --k) {
        // Bezpieczne sprawdzenie indeksu przed dostepem
        if (static_cast<size_t>(k) < dp.size() && dp[static_cast<size_t>(k)]) {
            best_sum_machine1 = k;
            break; // Znaleziono, mozna przerwać
        }
    }

    // Oblicz obciazenia obu maszyn i finalny Cmax
    long long load_machine1 = best_sum_machine1;
    long long load_machine2 = total_sum - best_sum_machine1; // Druga maszyna dostaje reszte
    machine_loads[0] = load_machine1; // Zapisz wynikowe obciazenia
    machine_loads[1] = load_machine2;
    return std::max(load_machine1, load_machine2); // Zwroc Cmax
}

// Algorytm Przegladu Zupelnego (Brute Force) dla P2||Cmax: Sprawdza wszystkie mozliwe sposoby przypisania kazdego zadania do jednej z dwoch maszyn.


long long brute_force_p2(int num_machines, const std::vector<int> &task_times, std::vector<long long> &machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm Brute Force jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size(); // Liczba zadan

    // Zmienne do sledzenia najlepszego znalezionego rozwiazania
    long long min_found_cmax = std::numeric_limits<long long>::max(); // Najlepszy Cmax
    long long optimal_load1 = -1, optimal_load2 = -1; // Obciazenia dla najlepszego Cmax

    long long num_assignments = 1LL << n;

    // ptetla iteruje przez wszystkie mozliwe przypisania.
    for (long long i = 0; i < num_assignments; ++i) {
        // Oblicz obciazenia dla przypisania 'i'
        long long current_load1 = 0; // Obciazenie maszyny 1 dla tego przypisania
        long long current_load2 = 0; // Obciazenie maszyny 2 dla tego przypisania

        // Petla po zadaniach (od j=0 do n-1)
        for (int j = 0; j < n; ++j) {
            // Sprawdzamy j-ty bit liczby i
            // (i >> j) przesuwa j-ty bit na pozycje 0
            // &1 izoluje ten bit z wynikiem 0 lub 1
            if (((i >> j) & 1) == 1) {
                // Jesli j-ty bit to 1 - przypisz j-te zadanie do maszyny 2
                current_load2 += task_times[j];
            } else {
                // Jesli j-ty bit to 0 - przypisz j-te zadanie do maszyny 1
                current_load1 += task_times[j];
            }
        }

        // Oblicz Cmax dla tego konkretnego przypisania
        long long current_cmax = std::max(current_load1, current_load2);

        // Jesli znaleziony Cmax jest lepszy
        if (current_cmax < min_found_cmax) {
            // zaktualizuj najlepszy Cmax i zapamietaj obciazenia dla tego rozwiazania
            min_found_cmax = current_cmax;
            optimal_load1 = current_load1;
            optimal_load2 = current_load2;
        }
    }

    // Zapisz wynikowe optymalne obciazenia
    if (optimal_load1 != -1) {
        machine_loads[0] = optimal_load1;
        machine_loads[1] = optimal_load2;
    } else if (n > 0)
        return -1;

    // Zwroć znaleziony minimalny Cmax (optymalny)
    return min_found_cmax;
}

// Algorytm PTAS dla P2||Cmax:
 //Sortuje zadania malejaco (jak LPT).
 // Pierwsze 'k' najdluzszych zadan przypisuje optymalnie uzywajac Brute Force
 // Pozostale zadania przypisuje jak LSA do maszyn juz obciazonych przez pierwsze 'k' zadan.

long long
ptas_p2(int num_machines, const std::vector<int> &task_times, int ptas_k, std::vector<long long> &machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm PTAS jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size();

    // Jesli k <= 0, nie ma sensu robić czesci optymalnej, uzyj LPT jako fallback
    if (ptas_k <= 0) {
        std::cerr << "Ostrzezenie PTAS: k <= 0, wykonuje LPT.\n";
        return longest_processing_time(num_machines, task_times, machine_loads);
    }

    // Rzeczywista liczba zadan rozpatrywanych optymalnie
    int k_actual = std::min(ptas_k, n);

    // Sortowanie zadan malejaco
    std::vector<int> sorted_tasks = task_times;
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    // Optymalne przypisanie pierwszych k_actual zadan
    long long min_partial_cmax = std::numeric_limits<long long>::max(); // Najlepszy Cmax dla pierwszych k zadan
    long long best_partial_load1 = 0; // Obciazenie m1 dla najlepszego Cmax
    long long best_partial_load2 = 0; // Obciazenie m2 dla najlepszego Cmax

    long long num_partial_assignments = 1LL << k_actual; // 2^k_actual

    // Petla Brute Force dla pierwszych k_actual zadan
    for (long long i = 0; i < num_partial_assignments; ++i) {
        long long current_load1 = 0, current_load2 = 0;
        // Dekodowanie przypisania 'i' dla zadan 0..k_actual-1
        for (int j = 0; j < k_actual; ++j) {
            if (((i >> j) & 1) == 1) current_load2 += sorted_tasks[j]; // Bit 1 -> M2
            else current_load1 += sorted_tasks[j];                     // Bit 0 -> M1
        }
        // Cmax dla tego czesciowego przypisania
        long long current_partial_cmax = std::max(current_load1, current_load2);
        // Jesli cmax jest lepszy
        if (current_partial_cmax < min_partial_cmax) {
            min_partial_cmax = current_partial_cmax; // aktualizuj najlepszy Cmax
            best_partial_load1 = current_load1;     // zapamietaj obciazenia
            best_partial_load2 = current_load2;
        }
    }

    // przypisanie pozostalych zadan (od k_actual do n-1)
    machine_loads[0] = best_partial_load1;
    machine_loads[1] = best_partial_load2;

    // Petla po pozostalych zadaniach
    for (int j = k_actual; j < n; ++j) {
        // Przypisz zadanie j do maszyny ktora aktualnie ma mniejsze obciazenie
        if (machine_loads[0] <= machine_loads[1]) {
            machine_loads[0] += sorted_tasks[j];
        } else {
            machine_loads[1] += sorted_tasks[j];
        }
    }

    // Zwroć finalny Cmax 
    return std::max(machine_loads[0], machine_loads[1]);
}

//Funkcja pomocnicza dla FPTAS: Rozwiazuje problem podzialu zbioru optymalnie dla (przeskalowanych) czasow zadan
// i odtwarza, ktore zadania trafily do pierwszej maszyny.


long long dp_for_fptas_with_backtracking(const std::vector<int> &scaled_tasks, long long target_sum_scaled,
                                         std::vector<int> &assignment_m1_indices) {
    int n = scaled_tasks.size(); // Liczba zadan
    assignment_m1_indices.clear(); // Wyczysc wektor wynikowy
    if (target_sum_scaled < 0)
        return 0;

    // Tworzenie tabeli DP: dp_table[i][k]
    // Wymiary: (n+1) wierszy (0..n zadan), (target_sum_scaled + 1) kolumn (sumy 0..target_sum_scaled)
    // dp_table[i][k] = true, jesli suma 'k' jest osiagalna przy uzyciu podzbioru pierwszych 'i' zadan.
    std::vector<std::vector<bool>> dp_table(n + 1,
                                            std::vector<bool>(static_cast<size_t>(target_sum_scaled) + 1, false));

    // Inicjalizacja
    dp_table[0][0] = true;

    // Wypelnianie tabeli DP - petla po zadaniach (wiersze)
    for (int i = 1; i <= n; ++i) {
        // Czas i-tego zadania
        long long current_scaled_task_time = static_cast<long long>(scaled_tasks[i - 1]);
        // Petla po mozliwych sumach
        for (long long k = 0; k <= target_sum_scaled; ++k) {
            // Suma 'k' byla osiagalna juz bez brania i-tego zadania.
            dp_table[i][k] = dp_table[i - 1][k];

            // Sprawdz czy mozemy osiagnać sume 'k' biorac i-te zadanie.
            if (current_scaled_task_time <= k && !dp_table[i][k]) { // Sprawdzamy tylko jesli jeszcze nie jest 'true'
                if (k - current_scaled_task_time >= 0 &&
                    dp_table[i - 1][static_cast<size_t>(k - current_scaled_task_time)]) {
                    dp_table[i][k] = true; // Suma k jest teraz osiagalna
                }
            }
        }
    }

    // Znajdź najlepsza osiagalna sume dla maszyny 1
    long long best_sum_scaled_m1 = 0;
    for (long long k = target_sum_scaled; k >= 0; --k) {
        if (dp_table[n][k]) {
            best_sum_scaled_m1 = k;
            break;
        }
    }

    // Backtracking
    long long current_sum_to_reconstruct = best_sum_scaled_m1;
    // Idziemy od ostatniego zadania  do pierwszego
    for (int i = n; i > 0 && current_sum_to_reconstruct > 0; --i) {
        // Sprawdzamy czy suma 'current_sum_to_reconstruct' byla osiagalna bez brania zadania i
        if (!dp_table[i - 1][static_cast<size_t>(current_sum_to_reconstruct)]) {
            int task_index = i - 1; // Indeks tego zadania w oryginalnej liscie
            assignment_m1_indices.push_back(task_index); // Dodaj indeks do wyniku

            // Zmniejsz sume o czas wzietego zadania
            long long task_time = static_cast<long long>(scaled_tasks[task_index]);
            current_sum_to_reconstruct -= task_time;
        }
    }
    // Odwracamy kolejnosć indeksow bo dodawalismy je idac od konca
    std::reverse(assignment_m1_indices.begin(), assignment_m1_indices.end());

    // Zwracamy najlepsza znaleziona sume dla maszyny 1
    return best_sum_scaled_m1;
}


// Algorytm FPTAS dla P2||Cmax:
//Oblicza wspolczynnik skalujacy 'K' na podstawie pozadanego bledu 'epsilon' i sumy czasow S.
//Skaluje czasy wszystkich zadan, dzielac je przez 'K' i biorac podloge ( p'_j = floor(p_j / K) ).
//Rozwiazuje problem P2||Cmax optymalnie dla przeskalowanych czasow p'_j, uzywajac DP z backtrackingiem,
//Stosuje uzyskane przypisanie do oryginalnych czasow zadan p_j, aby obliczyć przyblizony Cmax.

long long
fptas_p2(int num_machines, const std::vector<int> &task_times, double epsilon, std::vector<long long> &machine_loads) {

    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty())
        return 0;

    int n = task_times.size();

    // Skalowanie czasow, oblicz sume oryginalnych czasow (S)
    long long total_sum_original = 0;
    for (int duration: task_times) {
        total_sum_original += duration;
    }
    if (total_sum_original == 0)
        return 0; // Wszystkie czasy zerowe

    // Oblicz wspolczynnik skalujacy K (fptas_k_scale)
    // Formula: K = floor(epsilon * S / (2 * n))
    double scale_factor_calculated = std::floor(epsilon * static_cast<double>(total_sum_original) / (2.0 * n));
    long long fptas_k_scale = std::max(1LL, static_cast<long long>(scale_factor_calculated));

    //Utworz wektor przeskalowanych czasow p'_j = floor(p_j / K)
    std::vector<int> scaled_task_times(n);
    long long total_sum_scaled = 0; // Suma przeskalowanych czasow S'
    for (int i = 0; i < n; ++i) {
        // Dzielenie calkowite int / long long daje wynik floor
        scaled_task_times[i] = static_cast<int>(task_times[i] / fptas_k_scale);
        total_sum_scaled += scaled_task_times[i];
    }

    // Rozwiaz problem optymalnie dla przeskalowanych czasow p'_j, uzywamy DP z backtrackingiem
    long long target_sum_scaled = total_sum_scaled / 2; // Cel dla DP: floor(S'/2)
    std::vector<int> machine1_task_indices; // indeksy zadan dla maszyny 1

    // Wywolaj wewnetrzne DP z backtrackingiem
    dp_for_fptas_with_backtracking(scaled_task_times, target_sum_scaled, machine1_task_indices);

    // Zastosuj uzyskane przypisanie do oryginalnych czasow p_j
    long long final_load_m1 = 0; // Obciazenie maszyny 1 z oryginalnymi czasami
    long long final_load_m2 = 0; // Obciazenie maszyny 2 z oryginalnymi czasami
    std::vector<bool> assigned_to_m1_flag(n, false); // Flagi do sledzenia, ktore zadanie gdzie trafilo

    // Zsumuj oryginalne czasy dla zadan przypisanych do maszyny 1
    for (int index: machine1_task_indices) {
        if (index >= 0 && index < n) {
            final_load_m1 += task_times[index];
            assigned_to_m1_flag[index] = true; // Oznacz zadanie jako przypisane
        } else {
            throw std::runtime_error("FPTAS: Niepoprawny indeks zadania z backtrackingu DP.");
        }
    }

    // Zsumuj oryginalne czasy dla zadan nieprzypisanych do maszyny 1 (ida do maszyny 2)
    for (int i = 0; i < n; ++i) {
        if (!assigned_to_m1_flag[i]) {
            final_load_m2 += task_times[i];
        }
    }

    // Zapisz wynikowe obciazenia i zwroć przyblizony Cmax
    machine_loads[0] = final_load_m1;
    machine_loads[1] = final_load_m2;
    return std::max(final_load_m1, final_load_m2);
}


// Generuje wektor zadan z losowymi czasami wykonania

std::vector<int> generate_tasks(int num_tasks, int min_p, int max_p) {
    static std::random_device randomDevice;
    static std::mt19937 randomNumberEngine(randomDevice());
    std::uniform_int_distribution<int> distribution(min_p, max_p);
    std::vector<int> tasks(num_tasks);
    for (int i = 0; i < num_tasks; ++i)
        tasks[i] = distribution(randomNumberEngine);
    return tasks;
}

//Funkcja pomocnicza: Mierzy czas wykonania podanego algorytmu

std::pair<long long, long long> measure_execution(
        long long (*algorithm_function)(int, const std::vector<int> &, std::vector<long long> &),
        int num_machines,
        const std::vector<int> &task_times,
        std::vector<long long> &machine_loads) {
    long long cmax_result = -1;
    long long duration_result = -1;

    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        cmax_result = algorithm_function(num_machines, task_times, machine_loads);
        auto end_time = std::chrono::high_resolution_clock::now();


        if (cmax_result >= 0) {
            auto duration_object = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            duration_result = duration_object.count();
        } else {
            std::cerr << "Ostrzezenie: Algorytm zwrocil Cmax < 0.\n";
        }
    } catch (const std::exception &e) {
        std::cerr << "Wyjatek podczas wykonywania algorytmu: " << e.what() << std::endl;


        return {cmax_result, duration_result};
    }
}


// Funkcja pomocnicza: Formatuje wartosć Cmax lub czasu do stringa dla tabeli

std::string format_cell_output(long long value, int precision, const std::string &status_if_error = "ERROR") {
    std::stringstream cell_stream;
    if (value >= 0) {
        if (precision == 1) cell_stream << std::fixed << std::setprecision(1) << static_cast<double>(value);
        else cell_stream << value; // Czas w us jako liczba calkowita
    } else {
        cell_stream << status_if_error;
    }
    return cell_stream.str();
}

int main() {
    const int NUM_MACHINES = 2;
    const int PTAS_K_PARAM = 5;
    const double FPTAS_EPSILON_PARAM = 0.1;

    std::vector<InstanceConfig> test_configurations = {
            {10, 1, 10},
            {10, 10, 20},
            {15, 1, 10},
            {20, 1, 10},
            {20, 10, 20},

    };

    //formatowanie do tabeli wynikow
    const int COL_WIDTH_N = 7;
    const int COL_WIDTH_RANGE = 12;
    const int COL_WIDTH_CMAX = 14;
    const int COL_WIDTH_TIME = 17;
    const int TOTAL_TABLE_WIDTH = COL_WIDTH_N + COL_WIDTH_RANGE + 6 * (COL_WIDTH_CMAX + COL_WIDTH_TIME) + 15;

    std::cout << "\n--- Tabela Wynikow Eksperymentow (P2||Cmax) ---\n";
    std::cout << "Jeden przebieg na konfiguracje.\n";
    std::cout << "Parametry aproksymacyjne: PTAS k=" << PTAS_K_PARAM << ", FPTAS epsilon=" << FPTAS_EPSILON_PARAM
              << "\n\n";
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl;
    // nazwy kolumn
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

    // tesotownaie konfiguracji
    for (const InstanceConfig &config: test_configurations) {

        std::vector<int> current_tasks;
        // Generowanie zadan
        try {
            current_tasks = generate_tasks(config.num_tasks, config.min_task_time, config.max_task_time);
        } catch (const std::exception &e) {
            std::cerr << "KRYTYCZNY BLAD GENEROWANIA: " << e.what() << std::endl;
            std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                      << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";
            std::cout << std::right;
            for (int i = 0; i < 6; ++i) {
                std::cout << std::setw(COL_WIDTH_CMAX) << "CRITICAL" << "|" << std::setw(COL_WIDTH_TIME) << "CRITICAL"
                          << "|";
            }
            std::cout << std::endl;
            continue;
        }

        std::vector<long long> temp_loads;

        // LSA
        long long cmax_lsa = -1, time_lsa = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_lsa = list_scheduling_algorithm(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_lsa >= 0) time_lsa = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception &e) {}

        // LPT
        long long cmax_lpt = -1, time_lpt = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_lpt = longest_processing_time(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_lpt >= 0) time_lpt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception &e) {}

        // DP
        long long cmax_dp = -1, time_dp = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_dp = dynamic_programming_p2(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_dp >= 0) time_dp = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception &e) {}

        // BF
        long long cmax_bf = -1, time_bf = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_bf = brute_force_p2(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_bf >= 0) time_bf = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception &e) {}

        // PTAS
        long long cmax_ptas = -1, time_ptas = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_ptas = ptas_p2(NUM_MACHINES, current_tasks, PTAS_K_PARAM, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_ptas >= 0) time_ptas = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception &e) {}

        // FPTAS
        long long cmax_fptas = -1, time_fptas = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_fptas = fptas_p2(NUM_MACHINES, current_tasks, FPTAS_EPSILON_PARAM, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_fptas >= 0)
                time_fptas = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception &e) {}

        //wiersze wynikow
        std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                  << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";
        std::cout << std::right; // Wyrownanie do prawej dla wynikow

        //wyniki dla kazdego algorytmu
        std::cout << std::setw(COL_WIDTH_CMAX) << format_cell_output(cmax_lsa, 1, "FAIL") << "|"
                  << std::setw(COL_WIDTH_TIME) << format_cell_output(time_lsa, 0, "FAIL") << "|";
        std::cout << std::setw(COL_WIDTH_CMAX) << format_cell_output(cmax_lpt, 1, "FAIL") << "|"
                  << std::setw(COL_WIDTH_TIME) << format_cell_output(time_lpt, 0, "FAIL") << "|";
        std::cout << std::setw(COL_WIDTH_CMAX) << format_cell_output(cmax_dp, 1, "FAIL") << "|"
                  << std::setw(COL_WIDTH_TIME) << format_cell_output(time_dp, 0, "FAIL") << "|";
        std::cout << std::setw(COL_WIDTH_CMAX) << format_cell_output(cmax_bf, 1, "FAIL") << "|"
                  << std::setw(COL_WIDTH_TIME) << format_cell_output(time_bf, 0, "FAIL") << "|";
        std::cout << std::setw(COL_WIDTH_CMAX) << format_cell_output(cmax_ptas, 1, "FAIL") << "|"
                  << std::setw(COL_WIDTH_TIME) << format_cell_output(time_ptas, 0, "FAIL") << "|";
        std::cout << std::setw(COL_WIDTH_CMAX) << format_cell_output(cmax_fptas, 1, "FAIL") << "|"
                  << std::setw(COL_WIDTH_TIME) << format_cell_output(time_fptas, 0, "FAIL") << "|";

        std::cout << std::endl;


    }


    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl;
    std::cout << "Zakonczono testy\n" << std::endl;

    return 0;
}