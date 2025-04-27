#include <iostream> // Do obsługi wejścia/wyjścia (cout)
#include <vector>   // Do używania dynamicznych tablic (vector)
#include <numeric>  // Dla std::accumulate (sumowanie), std::max_element
#include <algorithm>// Dla std::sort, std::min_element, std::max_element, std::min, std::max
#include <limits>   // Dla std::numeric_limits (sprawdzanie zakresów typów)
#include <functional>// Dla std::greater (sortowanie malejąco)
#include <random>   // Do generowania liczb losowych
#include <chrono>   // Do pomiaru czasu
#include <iomanip>  // Do formatowania wyjścia (setw, setprecision, left, right, fixed)
#include <string>   // Do używania std::string
#include <cmath>    // Dla std::floor (chociaż nie jest krytyczne)
#include <stdexcept>// Do obsługi podstawowych wyjątków (np. invalid_argument)
#include <sstream>  // Dla std::stringstream (konwersja liczb na string, formatowanie)

// --- Definicje Struktur Pomocniczych ---

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
        ss << "[" << std::setw(3) << std::right << min_task_time << "-"
           << std::setw(3) << std::right << max_task_time << "]";
        return ss.str();
    }
};

// --- Implementacje Algorytmów i Funkcji Pomocniczych ---

/**
 * Funkcja pomocnicza: Znajduje indeks maszyny, która aktualnie
 * ma najmniejsze całkowite obciążenie (czas zakończenia pracy).
 * W przypadku remisu zwraca maszynę o niższym indeksie.
 *
 * @param machine_loads Wektor z aktualnymi czasami zakończenia pracy na każdej maszynie.
 * @return Indeks maszyny, która skończy najwcześniej. Zwraca -1 jeśli brak maszyn.
 */
int find_least_loaded_machine(const std::vector<long long>& machine_loads) {
    // Sprawdzenie, czy są jakieś maszyny
    if (machine_loads.empty()) {
        return -1; // Błąd
    }
    // Zakładamy, że pierwsza maszyna jest na razie najmniej obciążona
    int min_index = 0;
    long long min_load = machine_loads[0];
    // Przechodzimy przez pozostałe maszyny
    for (size_t i = 1; i < machine_loads.size(); ++i) {
        // Jeśli znaleźliśmy maszynę z mniejszym obciążeniem
        if (machine_loads[i] < min_load) {
            // Zapisujemy jej obciążenie i indeks
            min_load = machine_loads[i];
            min_index = static_cast<int>(i);
        }
    }
    // Zwracamy indeks maszyny, która zwolni się najwcześniej
    return min_index;
}

/**
 * Algorytm List Scheduling (LSA):
 * Prosta heurystyka zachłanna. Przechodzi przez zadania w podanej kolejności.
 * Każde zadanie przypisuje do maszyny, która aktualnie ma najmniejsze obciążenie
 * (czyli zwolni się najwcześniej).
 *
 * @param num_machines Liczba dostępnych identycznych maszyn (m).
 * @param task_times Wektor zawierający czasy wykonania poszczególnych zadań (p_j).
 * @param machine_loads (Parametr wyjściowy) Wektor, który zostanie wypełniony końcowymi
 *                      czasami zakończenia pracy na każdej maszynie.
 * @return Obliczony Cmax (maksymalny czas zakończenia pracy na dowolnej maszynie).
 *         Zwraca 0 dla pustych danych wejściowych.
 */
long long list_scheduling_algorithm(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    // Podstawowe sprawdzenia danych wejściowych
    if (num_machines <= 0) return 0;

    // Wyzerowanie obciążeń maszyn na początku
    machine_loads.assign(num_machines, 0LL); // Używa vector::assign do ustawienia rozmiaru i wartości

    // Jeśli nie ma zadań, Cmax wynosi 0
    if (task_times.empty()) return 0;

    // Główna pętla: przechodzi przez każde zadanie
    for (int duration : task_times) {
        // Znajdź maszynę, która skończy najwcześniej
        int target_machine = find_least_loaded_machine(machine_loads);
        // Przypisz bieżące zadanie do tej maszyny, zwiększając jej obciążenie
        // (Proste sprawdzenie, czy find_least_loaded_machine nie zwróciło błędu)
        if(target_machine != -1) {
            machine_loads[target_machine] += duration;
        } else {
            // Teoretycznie nie powinno się zdarzyć przy num_machines > 0
            return -1; // Zasygnalizuj wewnętrzny błąd
        }
    }

    // Po przypisaniu wszystkich zadań, Cmax to maksymalne obciążenie spośród wszystkich maszyn
    if (machine_loads.empty()) return 0; // Dodatkowe zabezpieczenie
    return *std::max_element(machine_loads.begin(), machine_loads.end()); // Używa std::max_element
}

/**
 * Algorytm Longest Processing Time First (LPT):
 * Ulepszona wersja LSA. Zanim zacznie przypisywać zadania,
 * sortuje je malejąco według czasów ich wykonania (najdłuższe najpierw).
 * Następnie stosuje tę samą logikę co LSA: przypisuje kolejne (posortowane)
 * zadanie do maszyny, która zwolni się najwcześniej.
 * Daje zazwyczaj znacznie lepsze wyniki niż LSA.
 *
 * @param num_machines Liczba maszyn (m).
 * @param task_times Wektor czasów wykonania zadań.
 * @param machine_loads (Wyjście) Wektor z końcowymi obciążeniami maszyn.
 * @return Obliczony Cmax.
 */
long long longest_processing_time(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    if (num_machines <= 0) return 0;
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    // 1. Utwórz kopię wektora czasów zadań
    std::vector<int> sorted_tasks = task_times;
    // 2. Posortuj kopię malejąco (najdłuższe zadania na początku)
    //    Używa std::sort i komparatora std::greater<int>()
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    // 3. Zastosuj algorytm LSA na posortowanej liście zadań
    //    Wywołujemy zaimplementowaną wcześniej funkcję LSA
    return list_scheduling_algorithm(num_machines, sorted_tasks, machine_loads);
}

/**
 * Algorytm Programowania Dynamicznego (DP) dla P2||Cmax:
 * Znajduje rozwiązanie optymalne dla problemu podziału zadań na DWIE maszyny.
 * Problem jest równoważny znalezieniu podzbioru zadań, którego suma czasów
 * jest jak najbliższa (ale nie większa) połowie sumy wszystkich czasów (S/2).
 * Używa tablicy jednowymiarowej do śledzenia osiągalnych sum (optymalizacja pamięci).
 * Złożoność czasowa: O(n * S), pamięciowa: O(S). Działa tylko dla m=2.
 *
 * @param num_machines Liczba maszyn (musi być 2).
 * @param task_times Wektor czasów wykonania zadań.
 * @param machine_loads (Wyjście) Wektor z optymalnymi obciążeniami maszyn.
 * @return Optymalny Cmax. Rzuca wyjątki przy błędach.
 */
long long dynamic_programming_p2(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    // Sprawdzenie, czy na pewno mamy 2 maszyny
    if (num_machines != 2) {
        // Rzucenie wyjątku, jeśli algorytm jest używany niepoprawnie
        throw std::invalid_argument("Algorytm DP jest zaimplementowany tylko dla m=2.");
    }
    // Inicjalizacja obciążeń dla 2 maszyn
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    // Krok 1: Oblicz całkowitą sumę czasów zadań (S)
    long long total_sum = 0;
    for (int duration : task_times) {
        // Podstawowe sprawdzenie poprawności danych
        if (duration < 0) throw std::invalid_argument("Czas zadania nie moze byc ujemny.");
        // Sprawdzenie, czy dodanie duration nie spowoduje przepełnienia long long
        if (total_sum > std::numeric_limits<long long>::max() - duration) {
            throw std::overflow_error("Przepełnienie podczas sumowania czasow zadan dla DP.");
        }
        total_sum += duration;
    }

    // Krok 2: Oblicz docelową sumę dla jednej maszyny (połowa sumy całkowitej)
    // Dzielenie całkowite automatycznie da floor(S/2)
    long long target_sum = total_sum / 2;

    // Krok 3: Stwórz tablicę dynamiczną 'dp'
    // dp[k] będzie miało wartość 'true', jeśli suma czasów 'k' jest możliwa do osiągnięcia
    // przez jakiś podzbiór zadań. Rozmiar to target_sum + 1 (indeksy 0 do target_sum).
    std::vector<bool> dp(static_cast<size_t>(target_sum) + 1, false);
    // Inicjalizacja: suma 0 jest zawsze możliwa (nie biorąc żadnego zadania)
    dp[0] = true;

    // Krok 4: Wypełnianie tablicy DP (optymalizacja pamięciowa - tylko jeden wiersz)
    long long max_reachable_sum_so_far = 0; // Pomocnicza zmienna do optymalizacji pętli wewnętrznej
    // Pętla po wszystkich zadaniach
    for (int duration : task_times) {
        // Ignorujemy zadania o czasie <= 0
        if (duration <= 0) continue;
        long long current_task_time = static_cast<long long>(duration);

        // Pętla wewnętrzna idzie OD TYŁU: od najwyższej możliwej sumy do czasu bieżącego zadania.
        // Idziemy od tyłu, aby uniknąć wielokrotnego użycia tego samego zadania w tym kroku.
        // Zaczynamy od min(max_reachable_sum_so_far + current_task_time, target_sum),
        // bo nie ma sensu sprawdzać sum większych niż target_sum, ani tych,
        // które na pewno nie mogą powstać przez dodanie current_task_time.
        for (long long k = std::min(max_reachable_sum_so_far + current_task_time, target_sum); k >= current_task_time; --k) {
            // Jeśli suma (k - czas bieżącego zadania) była osiągalna wcześniej...
            // (sprawdzamy dp[k - current_task_time])
            if (dp[static_cast<size_t>(k - current_task_time)]) {
                // ...to suma 'k' staje się teraz osiągalna.
                dp[static_cast<size_t>(k)] = true;
                // Aktualizujemy największą dotąd znalezioną osiągalną sumę
                max_reachable_sum_so_far = std::max(max_reachable_sum_so_far, k);
            }
        }
        // Szybkie sprawdzenie, czy przypadkiem nie osiągnęliśmy już celu
        if (dp[static_cast<size_t>(target_sum)]) {
            max_reachable_sum_so_far = target_sum;
        }
    }

    // Krok 5: Znajdź najlepszą (największą) sumę dla maszyny 1
    // Szukamy od target_sum w dół, aż znajdziemy pierwszą osiągalną sumę.
    long long best_sum_machine1 = 0;
    for (long long k = target_sum; k >= 0; --k) {
        // Bezpieczne sprawdzenie indeksu przed dostępem
        if (static_cast<size_t>(k) < dp.size() && dp[static_cast<size_t>(k)]) {
            best_sum_machine1 = k;
            break; // Znaleziono, można przerwać
        }
    }

    // Krok 6: Oblicz obciążenia obu maszyn i finalny Cmax
    long long load_machine1 = best_sum_machine1;
    long long load_machine2 = total_sum - best_sum_machine1; // Druga maszyna dostaje resztę
    machine_loads[0] = load_machine1; // Zapisz wynikowe obciążenia
    machine_loads[1] = load_machine2;
    return std::max(load_machine1, load_machine2); // Zwróć Cmax
}

/**
 * Algorytm Przeglądu Zupełnego (Brute Force) dla P2||Cmax:
 * Sprawdza *wszystkie* możliwe sposoby przypisania każdego zadania do jednej z dwóch maszyn.
 * Gwarantuje znalezienie optymalnego Cmax, ale jest bardzo wolny (złożoność O(2^n)).
 * Praktyczny tylko dla małej liczby zadań (n <= 20-25). Działa tylko dla m=2.
 *
 * @param num_machines Liczba maszyn (musi być 2).
 * @param task_times Wektor czasów wykonania zadań.
 * @param machine_loads (Wyjście) Wektor z optymalnymi obciążeniami maszyn.
 * @return Optymalny Cmax. Rzuca wyjątki przy błędach.
 */
long long brute_force_p2(int num_machines, const std::vector<int>& task_times, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm Brute Force jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size(); // Liczba zadań

    // Sprawdzenie, czy 'n' nie jest za duże dla operacji bitowych (przesunięcia)
    if (n >= 64) throw std::overflow_error("Liczba zadan (n=" + std::to_string(n) + ") zbyt duza dla operacji bitowych (>= 64).");
    // Sprawdzenie ujemnych czasów
    for (int duration : task_times) if (duration < 0) throw std::invalid_argument("Czas zadania nie moze byc ujemny.");

    // Zmienne do śledzenia najlepszego znalezionego rozwiązania
    long long min_found_cmax = std::numeric_limits<long long>::max(); // Najlepszy Cmax (inicjalizowany na max)
    long long optimal_load1 = -1, optimal_load2 = -1; // Obciążenia dla najlepszego Cmax

    // Liczba wszystkich możliwych przypisań zadań do 2 maszyn to 2 do potęgi n.
    // Używamy 1LL (long long), aby uniknąć przepełnienia przy przesunięciu bitowym dla n bliskiego 64.
    long long num_assignments = 1LL << n;

    // Główna pętla: iteruje przez wszystkie możliwe przypisania.
    // Liczba 'i' od 0 do 2^n - 1 reprezentuje jedno konkretne przypisanie.
    // Bity liczby 'i' mówią, do której maszyny trafi dane zadanie.
    for (long long i = 0; i < num_assignments; ++i) {
        // Oblicz obciążenia dla przypisania 'i'
        long long current_load1 = 0; // Obciążenie maszyny 1 dla tego przypisania
        long long current_load2 = 0; // Obciążenie maszyny 2 dla tego przypisania

        // Pętla po zadaniach (od j=0 do n-1)
        for (int j = 0; j < n; ++j) {
            // Sprawdzamy j-ty bit liczby 'i'.
            // (i >> j) przesuwa j-ty bit na pozycję 0.
            // & 1 izoluje ten bit (wynik to 0 lub 1).
            if (((i >> j) & 1) == 1) {
                // Jeśli j-ty bit to 1, przypisz j-te zadanie do maszyny 2
                current_load2 += task_times[j];
            } else {
                // Jeśli j-ty bit to 0, przypisz j-te zadanie do maszyny 1
                current_load1 += task_times[j];
            }
        }

        // Oblicz Cmax dla tego konkretnego przypisania
        long long current_cmax = std::max(current_load1, current_load2);

        // Jeśli znaleziony Cmax jest lepszy (mniejszy) niż dotychczasowy najlepszy...
        if (current_cmax < min_found_cmax) {
            // ...zaktualizuj najlepszy Cmax i zapamiętaj obciążenia dla tego rozwiązania
            min_found_cmax = current_cmax;
            optimal_load1 = current_load1;
            optimal_load2 = current_load2;
        }
    }

    // Zapisz wynikowe optymalne obciążenia (jeśli cokolwiek znaleziono)
    if (optimal_load1 != -1) {
        machine_loads[0] = optimal_load1;
        machine_loads[1] = optimal_load2;
    } else if (n > 0) return -1; // Błąd, powinno się coś znaleźć dla n>0

    // Zwróć znaleziony minimalny Cmax (optymalny)
    return min_found_cmax;
}

/**
 * Algorytm PTAS (Polynomial Time Approximation Scheme) dla P2||Cmax:
 * Algorytm aproksymacyjny, który pozwala kontrolować dokładność kosztem czasu.
 * 1. Sortuje zadania malejąco (jak LPT).
 * 2. Pierwsze 'k' najdłuższych zadań przypisuje optymalnie (używając Brute Force).
 * 3. Pozostałe zadania przypisuje zachłannie (jak LSA) do maszyn już obciążonych przez pierwsze 'k' zadań.
 * Im większe 'k', tym lepsza dokładność, ale krok 2 staje się wolniejszy (O(2^k)).
 *
 * @param num_machines Liczba maszyn (musi być 2).
 * @param task_times Wektor czasów zadań.
 * @param ptas_k Parametr 'k' kontrolujący dokładność (liczba zadań rozpatrywanych optymalnie).
 * @param machine_loads (Wyjście) Wektor z przybliżonymi obciążeniami maszyn.
 * @return Przybliżony Cmax. Rzuca wyjątki przy błędach.
 */
long long ptas_p2(int num_machines, const std::vector<int>& task_times, int ptas_k, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm PTAS jest zaimplementowany tylko dla m=2.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size();

    // Jeśli k <= 0, nie ma sensu robić części optymalnej, użyj LPT jako fallback
    if (ptas_k <= 0) {
        std::cerr << "Ostrzezenie PTAS: k <= 0, wykonuje LPT.\n";
        return longest_processing_time(num_machines, task_times, machine_loads);
    }

    // Rzeczywista liczba zadań rozpatrywanych optymalnie (nie więcej niż n)
    int k_actual = std::min(ptas_k, n);

    // Krok 1: Sortowanie zadań malejąco
    std::vector<int> sorted_tasks = task_times;
    std::sort(sorted_tasks.begin(), sorted_tasks.end(), std::greater<int>());

    // Krok 2: Optymalne przypisanie pierwszych k_actual zadań
    // Używamy logiki Brute Force na podproblemie z k_actual zadaniami.
    long long min_partial_cmax = std::numeric_limits<long long>::max(); // Najlepszy Cmax dla pierwszych k zadań
    long long best_partial_load1 = 0; // Obciążenie M1 dla najlepszego Cmax
    long long best_partial_load2 = 0; // Obciążenie M2 dla najlepszego Cmax

    // Sprawdzenie limitów dla k_actual (podobnie jak n w Brute Force)
    if (k_actual >= 64) throw std::overflow_error("Parametr k dla PTAS (k=" + std::to_string(k_actual) + ") zbyt duzy dla operacji bitowych (>= 64).");

    long long num_partial_assignments = 1LL << k_actual; // 2^k_actual

    // Pętla Brute Force dla pierwszych k_actual zadań
    for (long long i = 0; i < num_partial_assignments; ++i) {
        long long current_load1 = 0, current_load2 = 0;
        // Dekodowanie przypisania 'i' dla zadań 0..k_actual-1
        for (int j = 0; j < k_actual; ++j) {
            if (((i >> j) & 1) == 1) current_load2 += sorted_tasks[j]; // Bit 1 -> M2
            else current_load1 += sorted_tasks[j];                     // Bit 0 -> M1
        }
        // Cmax dla tego częściowego przypisania
        long long current_partial_cmax = std::max(current_load1, current_load2);
        // Jeśli jest lepszy niż dotychczasowy najlepszy...
        if (current_partial_cmax < min_partial_cmax) {
            min_partial_cmax = current_partial_cmax; // ...aktualizuj najlepszy Cmax
            best_partial_load1 = current_load1;     // ...i zapamiętaj obciążenia
            best_partial_load2 = current_load2;
        }
    }

    // Krok 3: Zachłanne przypisanie pozostałych zadań (od k_actual do n-1)
    // Zaczynamy od najlepszego obciążenia uzyskanego dla pierwszych k zadań
    machine_loads[0] = best_partial_load1;
    machine_loads[1] = best_partial_load2;

    // Pętla po pozostałych zadaniach
    for (int j = k_actual; j < n; ++j) {
        // Przypisz zadanie j do maszyny, która aktualnie ma MNIEJSZE obciążenie
        if (machine_loads[0] <= machine_loads[1]) {
            machine_loads[0] += sorted_tasks[j];
        } else {
            machine_loads[1] += sorted_tasks[j];
        }
    }

    // Zwróć finalny Cmax (maksymalne obciążenie po dodaniu wszystkich zadań)
    return std::max(machine_loads[0], machine_loads[1]);
}

/**
 * Funkcja pomocnicza dla FPTAS: Programowanie dynamiczne z backtrackingiem.
 * Rozwiązuje problem podziału zbioru optymalnie dla (przeskalowanych) czasów zadań
 * i odtwarza, które zadania trafiły do pierwszej maszyny.
 * Używa standardowej tabeli DP O(n*S'), gdzie S' to suma przeskalowanych czasów.
 *
 * @param scaled_tasks Wektor przeskalowanych czasów zadań.
 * @param target_sum_scaled Docelowa suma dla maszyny 1 (floor(S'/2)).
 * @param assignment_m1_indices (Wyjście) Wektor, do którego zostaną dodane indeksy
 *                              zadań (z oryginalnej listy) przypisanych do maszyny 1.
 * @return Najlepsza (największa <= target_sum_scaled) suma czasów osiągnięta dla maszyny 1.
 */
long long dp_for_fptas_with_backtracking(const std::vector<int>& scaled_tasks, long long target_sum_scaled, std::vector<int>& assignment_m1_indices) {
    int n = scaled_tasks.size(); // Liczba zadań
    assignment_m1_indices.clear(); // Wyczyść wektor wynikowy
    if (target_sum_scaled < 0) return 0; // Nic do zrobienia

    // Tworzenie tabeli DP: dp_table[i][k]
    // Wymiary: (n+1) wierszy (0..n zadań), (target_sum_scaled + 1) kolumn (sumy 0..target_sum_scaled)
    // dp_table[i][k] = true, jeśli suma 'k' jest osiągalna przy użyciu podzbioru pierwszych 'i' zadań.
    std::vector<std::vector<bool>> dp_table(n + 1, std::vector<bool>(static_cast<size_t>(target_sum_scaled) + 1, false));

    // Inicjalizacja: Suma 0 jest osiągalna z 0 zadań.
    dp_table[0][0] = true;

    // Wypełnianie tabeli DP - pętla po zadaniach (wiersze)
    for (int i = 1; i <= n; ++i) {
        // Czas i-tego zadania (pamiętaj, że indeksy zadań są 0..n-1, a wierszy 1..n)
        long long current_scaled_task_time = static_cast<long long>(scaled_tasks[i - 1]);
        // Pętla po możliwych sumach (kolumny)
        for (long long k = 0; k <= target_sum_scaled; ++k) {
            // Opcja 1: Suma 'k' była osiągalna już bez brania i-tego zadania.
            dp_table[i][k] = dp_table[i - 1][k];

            // Opcja 2: Sprawdź, czy możemy osiągnąć sumę 'k' biorąc i-te zadanie.
            //          Warunki: zadanie się mieści (current_scaled_task_time <= k)
            //                 ORAZ suma (k - czas zadania) była osiągalna w poprzednim kroku (i-1).
            if (current_scaled_task_time <= k && !dp_table[i][k]) { // Sprawdzamy tylko jeśli jeszcze nie jest 'true'
                if (k - current_scaled_task_time >= 0 && dp_table[i - 1][static_cast<size_t>(k - current_scaled_task_time)]) {
                    dp_table[i][k] = true; // Suma k jest teraz osiągalna
                }
            }
        }
    }

    // Znajdź najlepszą osiągalną sumę dla maszyny 1 (największa suma w ostatnim wierszu <= target_sum_scaled)
    long long best_sum_scaled_m1 = 0;
    for (long long k = target_sum_scaled; k >= 0; --k) {
        if (dp_table[n][k]) { best_sum_scaled_m1 = k; break; }
    }

    // Backtracking: Odtworzenie przypisania zadań do maszyny 1
    long long current_sum_to_reconstruct = best_sum_scaled_m1;
    // Idziemy od ostatniego zadania (i=n) do pierwszego
    for (int i = n; i > 0 && current_sum_to_reconstruct > 0; --i) {
        // Sprawdzamy, czy suma 'current_sum_to_reconstruct' była osiągalna *bez* brania zadania 'i'.
        // Jeśli NIE była osiągalna (dp_table[i-1][current_sum_to_reconstruct] jest false)...
        if (!dp_table[i - 1][static_cast<size_t>(current_sum_to_reconstruct)]) {
            // ...to znaczy, że zadanie 'i' MUSIAŁO zostać wzięte, aby osiągnąć tę sumę.
            int task_index = i - 1; // Indeks tego zadania w oryginalnej liście
            assignment_m1_indices.push_back(task_index); // Dodaj indeks do wyniku

            // Zmniejsz sumę, którą musimy jeszcze "złożyć", o czas wziętego zadania
            long long task_time = static_cast<long long>(scaled_tasks[task_index]);
            current_sum_to_reconstruct -= task_time;
        }
        // Jeśli suma była osiągalna bez zadania 'i', to znaczy, że go nie braliśmy (idzie do maszyny 2),
        // więc przechodzimy do poprzedniego zadania (i-1) bez zmiany current_sum_to_reconstruct.
    }
    // Odwracamy kolejność indeksów, bo dodawaliśmy je idąc od końca
    std::reverse(assignment_m1_indices.begin(), assignment_m1_indices.end());

    // Zwracamy najlepszą znalezioną sumę dla maszyny 1 (w przeskalowanym problemie)
    return best_sum_scaled_m1;
}


/**
 * Algorytm FPTAS (Fully Polynomial Time Approximation Scheme) dla P2||Cmax:
 * Algorytm aproksymacyjny o złożoności wielomianowej zarówno względem 'n' jak i 1/epsilon.
 * 1. Oblicza współczynnik skalujący 'K' na podstawie pożądanego błędu 'epsilon' i sumy czasów S.
 * 2. Skaluje czasy wszystkich zadań, dzieląc je przez 'K' i biorąc podłogę ( p'_j = floor(p_j / K) ).
 * 3. Rozwiązuje problem P2||Cmax optymalnie dla *przeskalowanych* czasów p'_j, używając DP z backtrackingiem,
 *    aby uzyskać przypisanie zadań (które zadania idą do maszyny 1, które do 2).
 * 4. Stosuje uzyskane przypisanie do *oryginalnych* czasów zadań p_j, aby obliczyć przybliżony Cmax.
 *
 * @param num_machines Liczba maszyn (musi być 2).
 * @param task_times Wektor oryginalnych czasów zadań.
 * @param epsilon Pożądany maksymalny błąd względny (np. 0.1 dla 10%). Musi być > 0.
 * @param machine_loads (Wyjście) Wektor z przybliżonymi obciążeniami maszyn.
 * @return Przybliżony Cmax. Rzuca wyjątki przy błędach.
 */
long long fptas_p2(int num_machines, const std::vector<int>& task_times, double epsilon, std::vector<long long>& machine_loads) {
    if (num_machines != 2) throw std::invalid_argument("Algorytm FPTAS jest zaimplementowany tylko dla m=2.");
    if (epsilon <= 0.0) throw std::invalid_argument("Parametr epsilon dla FPTAS musi byc wiekszy od 0.");
    machine_loads.assign(num_machines, 0LL);
    if (task_times.empty()) return 0;

    int n = task_times.size();

    // Krok 1: Skalowanie czasów
    // a) Oblicz sumę oryginalnych czasów (S)
    long long total_sum_original = 0;
    for (int duration : task_times) {
        if (duration < 0) throw std::invalid_argument("FPTAS: Czas zadania nie moze byc ujemny.");
        if (total_sum_original > std::numeric_limits<long long>::max() - duration) throw std::overflow_error("FPTAS: Przepełnienie podczas sumowania oryginalnych czasow.");
        total_sum_original += duration;
    }
    if (total_sum_original == 0) return 0; // Wszystkie czasy zerowe

    // b) Oblicz współczynnik skalujący K (fptas_k_scale)
    // Formuła: K = floor(epsilon * S / (2 * n))
    // Upewniamy się, że K >= 1, aby uniknąć dzielenia przez zero i zachować sens skalowania.
    double scale_factor_calculated = std::floor(epsilon * static_cast<double>(total_sum_original) / (2.0 * n));
    long long fptas_k_scale = std::max(1LL, static_cast<long long>(scale_factor_calculated));

    // c) Utwórz wektor przeskalowanych czasów p'_j = floor(p_j / K)
    std::vector<int> scaled_task_times(n);
    long long total_sum_scaled = 0; // Suma przeskalowanych czasów S'
    for (int i = 0; i < n; ++i) {
        // Dzielenie całkowite int / long long daje w C++ wynik floor dla dodatnich liczb
        scaled_task_times[i] = static_cast<int>(task_times[i] / fptas_k_scale);
        total_sum_scaled += scaled_task_times[i];
    }

    // Krok 2: Rozwiąż problem optymalnie dla przeskalowanych czasów p'_j
    // Używamy DP z backtrackingiem, aby dostać przypisanie zadań.
    long long target_sum_scaled = total_sum_scaled / 2; // Cel dla DP: floor(S'/2)
    std::vector<int> machine1_task_indices; // Tu trafią indeksy zadań dla maszyny 1

    // Wywołaj wewnętrzne DP z backtrackingiem
    dp_for_fptas_with_backtracking(scaled_task_times, target_sum_scaled, machine1_task_indices);

    // Krok 3: Zastosuj uzyskane przypisanie do ORYGINALNYCH czasów p_j
    long long final_load_m1 = 0; // Obciążenie maszyny 1 z oryginalnymi czasami
    long long final_load_m2 = 0; // Obciążenie maszyny 2 z oryginalnymi czasami
    std::vector<bool> assigned_to_m1_flag(n, false); // Flagi do śledzenia, które zadanie gdzie trafiło

    // Zsumuj oryginalne czasy dla zadań przypisanych do maszyny 1
    for (int index : machine1_task_indices) {
        // Proste sprawdzenie poprawności indeksu zwróconego przez DP
        if (index >= 0 && index < n) {
            final_load_m1 += task_times[index];
            assigned_to_m1_flag[index] = true; // Oznacz zadanie jako przypisane
        } else {
            throw std::runtime_error("FPTAS: Niepoprawny indeks zadania z backtrackingu DP.");
        }
    }

    // Zsumuj oryginalne czasy dla zadań nieprzypisanych do maszyny 1 (idą do maszyny 2)
    for (int i = 0; i < n; ++i) {
        if (!assigned_to_m1_flag[i]) {
            final_load_m2 += task_times[i];
        }
    }

    // Zapisz wynikowe obciążenia i zwróć przybliżony Cmax
    machine_loads[0] = final_load_m1;
    machine_loads[1] = final_load_m2;
    return std::max(final_load_m1, final_load_m2);
}

/**
 * Generuje wektor zadań z losowymi czasami wykonania.
 */
std::vector<int> generate_tasks(int num_tasks, int min_p, int max_p) {
    if (num_tasks < 0 || min_p < 0 || max_p < min_p) throw std::invalid_argument("Nieprawidlowe argumenty dla generate_tasks.");
    // Używamy static, aby generator i rozkład były inicjalizowane tylko raz
    static std::random_device randomDevice;
    static std::mt19937 randomNumberEngine(randomDevice());
    // Tworzymy dystrybucję *wewnątrz* funkcji, aby reagowała na zmiany min_p/max_p
    std::uniform_int_distribution<int> distribution(min_p, max_p);
    std::vector<int> tasks(num_tasks);
    for (int i = 0; i < num_tasks; ++i) tasks[i] = distribution(randomNumberEngine);
    return tasks;
}

/**
 * Funkcja pomocnicza: Mierzy czas wykonania podanego algorytmu.
 * UWAGA: Usunięto rozbudowaną obsługę błędów dla uproszczenia.
 * Zakłada, że algorytm albo się powiedzie i zwróci Cmax >= 0,
 * albo rzuci wyjątek, który zostanie złapany w main.
 *
 * @param algorithm_function Wskaźnik do funkcji algorytmu.
 * @param num_machines Liczba maszyn.
 * @param task_times Wektor czasów zadań.
 * @param machine_loads Wektor na wynikowe obciążenia (modyfikowany przez algorytm).
 * @return Para: <Cmax, Czas wykonania w mikrosekundach>. Cmax=-1 jeśli błąd.
 */
std::pair<long long, long long> measure_execution(
        long long (*algorithm_function)(int, const std::vector<int>&, std::vector<long long>&),
        int num_machines,
        const std::vector<int>& task_times,
        std::vector<long long>& machine_loads)
{
    long long cmax_result = -1;
    long long duration_result = -1;

    // Używamy standardowego bloku try-catch do łapania wyjątków z algorytmów
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        cmax_result = algorithm_function(num_machines, task_times, machine_loads);
        auto end_time = std::chrono::high_resolution_clock::now();

        // Jeśli algorytm zakończył się poprawnie (nie rzucił wyjątku)
        // i zwrócił sensowny Cmax (>=0)
        if (cmax_result >= 0) {
            auto duration_object = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            duration_result = duration_object.count();
        } else {
            // Algorytm zwrócił błąd (np. -1), ale nie rzucił wyjątku
            std::cerr << "Ostrzezenie: Algorytm zwrocil Cmax < 0.\n";
            // Pozostawiamy cmax_result = -1, duration_result = -1
        }
    } catch (const std::exception& e) {
        // Złapano wyjątek rzucony przez algorytm
        std::cerr << "Wyjatek podczas wykonywania algorytmu: " << e.what() << std::endl;
        // Pozostawiamy cmax_result = -1, duration_result = -1
    } catch (...) {
        // Złapano nieznany wyjątek
        std::cerr << "Nieznany wyjatek podczas wykonywania algorytmu." << std::endl;
        // Pozostawiamy cmax_result = -1, duration_result = -1
    }

    return {cmax_result, duration_result};
}

/**
 * Funkcja pomocnicza: Formatuje wartość Cmax lub czasu do stringa dla tabeli.
 */
std::string format_cell_output(long long value, int precision, const std::string& status_if_error = "ERROR") {
    std::stringstream cell_stream;
    if (value >= 0) { // Poprawna wartość
        if (precision == 1) cell_stream << std::fixed << std::setprecision(1) << static_cast<double>(value);
        else cell_stream << value; // Czas w us jako liczba całkowita
    } else {
        cell_stream << status_if_error; // Wypisz status błędu
    }
    return cell_stream.str();
}

// --- Główna Funkcja Programu ---
int main() {
    // --- Konfiguracja Eksperymentu ---
    const int NUM_MACHINES = 2;
    const int PTAS_K_PARAM = 5;       // Stały parametr 'k' dla PTAS
    const double FPTAS_EPSILON_PARAM = 0.1; // Stały parametr 'epsilon' dla FPTAS

    // Lista konfiguracji instancji do przetestowania
    std::vector<InstanceConfig> test_configurations = {
            {10, 1, 10}, {10, 10, 20},
            {15, 1, 10},
            {20, 1, 10}, {20, 10, 20},
            // {25, 1, 5},   // Może być bardzo wolne dla BF/PTAS
            // {50, 50, 100}, // Może powodować błędy pamięci DP/FPTAS
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
    // Nazwy kolumn
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
        // Generowanie zadań - prosty try-catch na wypadek błędu generatora
        try {
            current_tasks = generate_tasks(config.num_tasks, config.min_task_time, config.max_task_time);
        } catch (const std::exception& e) {
            std::cerr << "KRYTYCZNY BLAD GENEROWANIA: " << e.what() << std::endl;
            // Wypisz wiersz z błędami i kontynuuj
            std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                      << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";
            std::cout << std::right;
            for(int i=0; i<6; ++i) { std::cout << std::setw(COL_WIDTH_CMAX) << "CRITICAL" << "|" << std::setw(COL_WIDTH_TIME) << "CRITICAL" << "|"; }
            std::cout << std::endl;
            continue;
        }

        std::vector<long long> temp_loads; // Pomocniczy wektor na obciążenia

        // Wywołania algorytmów i pomiar czasu - teraz bezpośrednio
        // Używamy std::chrono do pomiaru czasu wokół każdego wywołania

        // LSA
        long long cmax_lsa = -1, time_lsa = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_lsa = list_scheduling_algorithm(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_lsa >= 0) time_lsa = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) { /* Ignorujemy szczegóły błędu dla uproszczenia */ }

        // LPT
        long long cmax_lpt = -1, time_lpt = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_lpt = longest_processing_time(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_lpt >= 0) time_lpt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) { /* Ignorujemy */ }

        // DP
        long long cmax_dp = -1, time_dp = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_dp = dynamic_programming_p2(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_dp >= 0) time_dp = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) { /* Ignorujemy */ }

        // BF
        long long cmax_bf = -1, time_bf = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_bf = brute_force_p2(NUM_MACHINES, current_tasks, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_bf >= 0) time_bf = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) { /* Ignorujemy */ }

        // PTAS
        long long cmax_ptas = -1, time_ptas = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_ptas = ptas_p2(NUM_MACHINES, current_tasks, PTAS_K_PARAM, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_ptas >= 0) time_ptas = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) { /* Ignorujemy */ }

        // FPTAS
        long long cmax_fptas = -1, time_fptas = -1;
        try {
            auto start = std::chrono::high_resolution_clock::now();
            cmax_fptas = fptas_p2(NUM_MACHINES, current_tasks, FPTAS_EPSILON_PARAM, temp_loads);
            auto end = std::chrono::high_resolution_clock::now();
            if (cmax_fptas >= 0) time_fptas = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        } catch (const std::exception& e) { /* Ignorujemy */ }

        // --- Drukowanie Wiersza Wyników dla Konfiguracji ---
        std::cout << std::left << "|" << std::setw(COL_WIDTH_N) << config.num_tasks << "|"
                  << std::setw(COL_WIDTH_RANGE) << config.range_string_formatted() << "|";
        std::cout << std::right; // Wyrównanie do prawej dla wyników

        // Drukowanie wyników dla każdego algorytmu
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

        std::cout << std::endl; // Koniec wiersza tabeli

        // Usunięto drukowanie szczegółów błędów

    } // Koniec pętli po konfiguracjach (config)

    // --- Drukowanie Stopki Tabeli ---
    std::cout << std::string(TOTAL_TABLE_WIDTH, '-') << std::endl; // Linia dolna
    std::cout << "Koniec eksperymentow.\n" << std::endl;

    return 0; // Zakończ program pomyślnie
}