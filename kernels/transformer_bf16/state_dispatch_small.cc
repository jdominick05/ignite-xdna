// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// 2192-word packets; GELU table lives in the initialized activation arena.
#define BF16_SMALL_PACKETS
#define BF16_ARENA_UPLOAD_MAX 4352
#include "state_dispatch_exact.cc"
