#include "VariableRegistry.h"

DeviceVariables& GetGlobalRegistry() {
    static DeviceVariables registry{};
    return registry;
}