# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BINARY_7Z "OFF")
set(CPACK_BINARY_IFW "OFF")
set(CPACK_BINARY_NSIS "ON")
set(CPACK_BINARY_NUGET "OFF")
set(CPACK_BINARY_WIX "OFF")
set(CPACK_BINARY_ZIP "OFF")
set(CPACK_BUILD_SOURCE_DIRS "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master;C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build")
set(CPACK_CMAKE_GENERATOR "Visual Studio 17 2022")
set(CPACK_COMPONENTS_ALL "Unspecified;development;library")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "C:/Program Files/CMake/share/cmake-3.23/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "PolyVox built using CMake")
set(CPACK_GENERATOR "NSIS")
set(CPACK_INSTALL_CMAKE_PROJECTS "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build;PolyVox;ALL;/")
set(CPACK_INSTALL_PREFIX "C:/Program Files (x86)/PolyVox")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_CONTACT "matt@milliams.com")
set(CPACK_NSIS_DISPLAY_NAME "PolyVox SDK 0.2.1")
set(CPACK_NSIS_DISPLAY_NAME_SET "TRUE")
set(CPACK_NSIS_HELP_LINK "http:\\\\thermite3d.org/phpBB/")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
set(CPACK_NSIS_MODIFY_PATH "ON")
set(CPACK_NSIS_PACKAGE_NAME "PolyVox SDK 0.2.1")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\thermite3d.org")
set(CPACK_OUTPUT_CONFIG_FILE "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "C:/Program Files/CMake/share/cmake-3.23/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "PolyVox SDK")
set(CPACK_PACKAGE_FILE_NAME "PolyVox SDK-0.2.1-win64")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "PolyVox SDK 0.2.1")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "PolyVox SDK 0.2.1")
set(CPACK_PACKAGE_NAME "PolyVox SDK")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "Thermite 3D Team")
set(CPACK_PACKAGE_VERSION "0.2.1")
set(CPACK_PACKAGE_VERSION_MAJOR "0")
set(CPACK_PACKAGE_VERSION_MINOR "2")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_RESOURCE_FILE_LICENSE "C:/Program Files/CMake/share/cmake-3.23/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "C:/Program Files/CMake/share/cmake-3.23/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "C:/Program Files/CMake/share/cmake-3.23/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_7Z "ON")
set(CPACK_SOURCE_GENERATOR "7Z;ZIP")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/CPackSourceConfig.cmake")
set(CPACK_SOURCE_ZIP "ON")
set(CPACK_SYSTEM_NAME "win64")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "win64")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()

# Configuration for component "library"

SET(CPACK_COMPONENTS_ALL Unspecified development library)
set(CPACK_COMPONENT_LIBRARY_DISPLAY_NAME "Library")
set(CPACK_COMPONENT_LIBRARY_DESCRIPTION "The runtime libraries")
set(CPACK_COMPONENT_LIBRARY_REQUIRED TRUE)

# Configuration for component "development"

SET(CPACK_COMPONENTS_ALL Unspecified development library)
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Development")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Files required for developing with PolyVox")
set(CPACK_COMPONENT_DEVELOPMENT_DEPENDS library)

# Configuration for component "example"

SET(CPACK_COMPONENTS_ALL Unspecified development library)
set(CPACK_COMPONENT_EXAMPLE_DISPLAY_NAME "OpenGL Example")
set(CPACK_COMPONENT_EXAMPLE_DESCRIPTION "A PolyVox example application using OpenGL")
set(CPACK_COMPONENT_EXAMPLE_DEPENDS library)

# Configuration for component group "bindings"
set(CPACK_COMPONENT_GROUP_BINDINGS_DISPLAY_NAME "Bindings")
set(CPACK_COMPONENT_GROUP_BINDINGS_DESCRIPTION "Language bindings")

# Configuration for component "python"

SET(CPACK_COMPONENTS_ALL Unspecified development library)
set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME "Python Bindings")
set(CPACK_COMPONENT_PYTHON_DESCRIPTION "PolyVox bindings for the Python language")
set(CPACK_COMPONENT_PYTHON_GROUP bindings)
set(CPACK_COMPONENT_PYTHON_DEPENDS library)
set(CPACK_COMPONENT_PYTHON_DISABLED TRUE)
