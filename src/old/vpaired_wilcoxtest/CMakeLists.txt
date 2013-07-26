PROJECT(vpaired_wilcoxtest)

ADD_EXECUTABLE(vpaired_wilcoxtest vpaired_wilcoxtest.c)
TARGET_LINK_LIBRARIES(vpaired_wilcoxtest lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vpaired_wilcoxtest
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
