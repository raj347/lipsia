PROJECT(vtrialaverage)

INCLUDE_DIRECTORIES(../lib_lipsia)

ADD_EXECUTABLE(vtrialaverage vtrialaverage.c)
TARGET_LINK_LIBRARIES(vtrialaverage lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vtrialaverage
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
