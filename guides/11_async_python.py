#!/usr/bin/env python
# coding: utf-8

# # Async Python
#
# ## A briefing on asynchronous python coding, essential in Agent engineering

# Here is a masterful tutorial by you-know-who with exercises and comparisons.
#
# https://chatgpt.com/share/680648b1-b0a0-8012-8449-4f90b540886c
#
# This includes how to run async code from a python module.
#
# ### And now some examples:

# %%:


# Let's define an async function

import asyncio


async def do_some_work():
    print("Starting work")
    await asyncio.sleep(1)
    print("Work complete")


# %%:


# What will this do?

do_some_work()  # type: ignore[unused-coroutine]  # intentionally not awaited to demonstrate the behaviour


# %%:


# OK let's try that again!

asyncio.run(do_some_work())


# %%:


# What's wrong with this?


async def do_a_lot_of_work_v1():
    do_some_work()  # type: ignore[unused-coroutine]  # intentionally missing await to demonstrate the problem
    do_some_work()  # type: ignore[unused-coroutine]
    do_some_work()  # type: ignore[unused-coroutine]


asyncio.run(do_a_lot_of_work_v1())


# %%:


# Interesting warning! Let's fix it


async def do_a_lot_of_work():
    await do_some_work()
    await do_some_work()
    await do_some_work()


asyncio.run(do_a_lot_of_work())


# %%:


# And now let's do it in parallel
# It's important to recognize that this is not "multi-threading" in the way that you may be used to
# The asyncio library is running on a single thread, but it's using a loop to switch between tasks while one is waiting


async def do_a_lot_of_work_in_parallel():
    await asyncio.gather(do_some_work(), do_some_work(), do_some_work())


asyncio.run(do_a_lot_of_work_in_parallel())


# ### Finally - try writing a python module that calls do_a_lot_of_work_in_parallel
#
# See the link at the top; you'll need something like this in your module:
#
# ```python
# if __name__ == "__main__":
#     asyncio.run(do_a_lot_of_work_in_parallel())
# ```
