CC ?= cc
PKG_CONFIG ?= pkg-config
CFLAGS ?= -O2 -Wall -Wextra -Werror
CPPFLAGS += -DCKE_REVISION='"$(shell git rev-parse HEAD)"'
LDLIBS += $(shell $(PKG_CONFIG) --libs libzmq json-c libcurl)
CPPFLAGS += $(shell $(PKG_CONFIG) --cflags libzmq json-c libcurl)
node-adapter: node_adapter.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@
clean:
	rm -f node-adapter
.PHONY: clean
