//
// Created by clementzhao on 20/11/2026.
//

#include <iostream>

int main() {
    std::cout << "Hello World!" << std::endl;
    int n = 2;
    float e = 2.77374;
    std::cout << "value of N: " << n << "\nvalue of e: " << e  << std::endl;


    std::string name;
    std::cout << "whats ur name?: ";
    std::cin >> name;
    std::cout << "ur name is: IDIOT " << name << " Nigger\n";

    std::string fullName;
    std::getline(std::cin >> std::ws, fullName);

    return 0;
}
