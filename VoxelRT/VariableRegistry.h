#pragma once

#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <functional>
#include <stdexcept>

class DeviceVariables
{
public:
    using Setter = std::function<void(const void*)>;

    class VariableSetCache
    {
    private:
        Setter setter;
        DeviceVariables* const parent;

    public:
        VariableSetCache(Setter setter, DeviceVariables* parent)
            : setter(std::move(setter)), parent(parent)
        {
        }

        template<typename T>
        DeviceVariables& set(const T& value)
        {
            setter(&value);
            return *parent;
        }
    };

    template<typename T>
    VariableSetCache registerVariable(
        const std::string& name,
        T& deviceVar)
    {
        auto res = setters.try_emplace(
            name,
            [&deviceVar](const void* value)
            {
                cudaMemcpyToSymbol(
                    deviceVar,
                    value,
                    sizeof(T)
                );
            }
        );

        return VariableSetCache(res.first->second, this);
    }

    template<typename T>
    DeviceVariables& set(
        const std::string& name,
        const T& value)
    {
        auto it = setters.find(name);

        if (it == setters.end())
            throw std::runtime_error(
                "Unknown device variable: " + name
            );

        it->second(&value);

        return *this;
    }

private:
    std::unordered_map<std::string, Setter> setters;
};

DeviceVariables& GetGlobalRegistry();

#define DEVICE_VARIABLE(type, name)                                 \
    __device__ type name;                                           \
    namespace {                                                     \
        struct AutoRegister_##name                                  \
        {                                                           \
            AutoRegister_##name()                                   \
            {                                                       \
                GetGlobalRegistry().registerVariable(#name, name);  \
            }                                                       \
        };                                                          \
        static AutoRegister_##name                                  \
            autoRegister_##name;                                    \
    }