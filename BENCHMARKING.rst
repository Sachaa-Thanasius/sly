How to Run
==========

**Note: This is very informal and more for me than anyone else.**


Setup
-----

Install yelp-gprof2dot:

.. code-block:: sh

    python -m pip install -U yelp-gprof2dot


Bench Table Creation
--------------------

Run some variation of the following:

.. code-block:: sh

    python -m cProfile -o import_log.pstats -m examples.c11.parser run
    gprof2dot import_log.pstats -z "yacc:2088:__init_subclass__" | dot -Tsvg -o import_log.svg
    firefox import_log.svg  # Or view it some other way.


Bench Runtime
-------------

Run some variation of the following:

.. code-block:: sh

    python -m cProfile -o run_log.pstats -m examples.json run
    gprof2dot run_log.pstats -z "yacc:2293:parse" | dot -Tsvg -o run_log.svg
    firefox run_log.svg  # Or view it some other way.
