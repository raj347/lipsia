PROJECT(vaddcovariates)

ADD_EXECUTABLE(vaddcovariates vaddcovariates.c)
TARGET_LINK_LIBRARIES(vaddcovariates lipsia ${VIA_LIBRARY})

SET_TARGET_PROPERTIES(vaddcovariates
                      PROPERTIES COMPILE_FLAGS -ansi)

INSTALL(TARGETS vaddcovariates
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
