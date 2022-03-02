# Install script for directory: C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/library/PolyVoxCore

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/PolyVox")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibraryx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/library/PolyVoxCore/Debug/PolyVoxCore.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/library/PolyVoxCore/Release/PolyVoxCore.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/library/PolyVoxCore/MinSizeRel/PolyVoxCore.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore/lib" TYPE STATIC_LIBRARY FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/library/PolyVoxCore/RelWithDebInfo/PolyVoxCore.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevelopmentx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore" TYPE DIRECTORY FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/library/PolyVoxCore/include" REGEX "/[^/]*\\.svn[^/]*$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore/lib" TYPE FILE FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/library/PolyVoxCore/Debug/PolyVoxCore.pdb")
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/PolyVoxCore/lib" TYPE FILE FILES "C:/Users/jimiu/Desktop/INFOMCV-Assignment-2/PolyVox-master/build/library/PolyVoxCore/RelWithDebInfo/PolyVoxCore.pdb")
  endif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
endif()

