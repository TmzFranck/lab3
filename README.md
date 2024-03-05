# Praktikumsprojekt Parallel-Programming(SPP)

Willkommen zu meinem Praktikumsprojekt! In diesem Repository findest du alle relevanten Informationen und Ressourcen.

## Projektbeschreibung

Alle Algorithmen der lab2 auf der GPU  implementieren

## Installation

1. **Voraussetzungen**: Stelle sicher, dass du die erforderlichen Tools und Bibliotheken installiert hast.
Dieses und die folgenden Praktika benötigen Softwarepakete, mit denen Sie gegebenenfalls noch nicht gearbeitet haben:
CMake, Make, git, einen modernen C/C++-Kompiler, später CUDA.
Auf dem Lichtenberg-Hochleistungsrechner erhalten Sie diese mit: module load git cmake gcc/11.
Falls Sie eine sinnvolle Linuxdistribution verwenden, können Sie diese installieren mit: sudo apt install git
cmake binutils g++-11 gcc-11.
Falls Sie Windows verwenden, empfehlen wir Visual Studio 2022, ggf. brauchen Sie CMake und git extra.
3. **Klonen des Repositories**: Führe den folgenden Befehl aus, um das Repository auf deinem lokalen System zu klonen:
- git clone https://github.com/TmzFranck/lab3.git



4. **Setup**: Gehe in das Projektverzeichnis und führe die notwendigen Setup-Schritte aus.
Wenn Sie eine sinnvolle Linuxdistribution verwenden (wie auf dem Lichtenberg), navigieren Sie im Terminal in den
entpackten Ordner und führen aus: mkdir build && cd build && cmake .. && make -j. Damit erstellen Sie
den Ordner, in dem das Programm gebaut wird, konfigurieren das Projekt (ggf. wiederholen, s.u.) und kompilieren
Ihren Quellcode (ggf. wiederholen, s.u.).
Wenn Sie Windows benutzen, erstellen Sie ein neues VS-Projekt (z.B. über die CMake-GUI) und wählen als Ordner,
in dem die Binärdateien kompiliert werden, auch build o.ä. aus. In der CMake-GUI klicken Sie nacheinander auf
Konfigurieren, Generieren, Projekt öffnen.

## Verwendung

Die CMake-Dateien sind für die Struktur Ihres Projektes zuständig. Darin sind mehrere Projekte definiert:
- lab_lib: Darin entwickeln Sie das ganze eigentliche Projekt. Die entsprechende CMake-Datei ist
/source/CMakeLists.txt. Falls Sie weitere .cpp-Dateien zum Projekt hinzufügen möchten, müssen Sie diese dort in
die Liste eintragen.
- lab: Das Projekt ist beinhaltet nur den Haupteinsprungspunkt des Programms (die main-Funktion). Darin
verarbeiten Sie die Kommandozeilenargumente und rufen die Funktionen Ihres Programms auf.
- lab_benchmarks: Das Projekt verwendet das Framework von Google für micro-benchmarking. Die entsprechende
CMake-Datei ist /benchmark/CMakeLists.txt.
- lab_test: Das Projekt verwendet das Framework von Google für Unittests. Die entsprechende CMake-Datei ist
/test/CMakeLists.txt
- /cmake/...: Der Ordner beinhaltet mehrere Skriptdateien, die Sie eigentlich nicht ändern müssen. Falls Sie Visual
Studio benutzen und dort Ihre .h-Dateien nicht angezeigt werden, fügen Sie diese in /cmake/VisualStudio.cmake
hinzu.




##  Hochladen des Quellcodes

Zippen Sie:
- Den Ordner /cmake/
- Den Ordner /source/
- Die Datei CMakeLists.txt


