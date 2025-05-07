#pragma once

#include <string>

namespace pccl {
namespace profile {

void internalStartProfile(const char* eventName, const char* extraInfo);
void internalEndProfile(const char* eventName);
void exportProfile(const std::string& exportDir);
void clearProfileData();

}  // namespace profile
}  // namespace pccl

#define PROFILE_START(eventName, extraInfo) \
  ::pccl::profile::internalStartProfile(eventName, extraInfo)
#define PROFILE_END(eventName) ::pccl::profile::internalEndProfile(eventName)
#define PROFILE_EXPORT(exportDir) ::pccl::profile::exportProfile(exportDir)
#define PROFILE_CLEAR() ::pccl::profile::clearProfileData()
