.. _sec_how_to_make_a_new_release:

====================================
How to publish a new release of NNMT
====================================

1.  Checkout the develop branch

    .. code::

        git checkout develop
        git pull


2.  Check that the code is working

    Make sure the new feature you implemented does what it is supposed to do.

3.  Check that code is tested

    The point above should be done by writing tests that ensure this.

4.  Check that code is well documented

    Follow the guidelines in the docs for this.

5.  Check the release notes

    All changes need to be documented in the release notes.

6.  Check that all tests run

    .. code::

        cd tests/
        pytest
        cd ../

7.  Create a release branch

    .. code::

        git checkout -b release-<release_number>


8.  Fix anything left

9.  Update version numbers to pre-format

    In ``nnmt/__init__.py``, ``docs/source/conf.py``, and ``setup.py``.

    **WARNING**: You can upload each file only once to testpypi, even if you delete them on testpypi. Therefore use version names like ``1.0.0a0``.

10. Create new conda environment and check local installation

    .. code::

        conda create -n temp0
        conda activate temp0
        conda install setuptools
        pip install .
        cd tests/
        pytest
        cd ../

11. Create sdist and wheel

    Remove any old wheels

    .. code::

        rm -r dist/

    Then check that ``setup.py`` contains all necessary data by running

    .. code::

        python setup.py check

    If this works, you can create the sdist and wheel source distribution using

    .. code::

        python setup.py sdist bdist_wheel

    which creates a new directory ``dist``.

12. Upload to test.pypi

    Then install ``twine`` and upload to testpypi

    **WARNING**: You can upload each file only once, even if you delete them on
    testpypi. Therefore use version names like ``1.0.0a``.

    .. code::

        conda install twine
        twine upload --repository-url https://test.pypi.org/legacy/ dist/*

13. Create new conda env and test install from test.pyi

    .. code::

        conda create -n temp1
        conda activate temp1
        conda install pip
        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nnmt==<release_number>
        cd tests/
        pytest
        cd ../

14. If this fails, fix problems, change version numbers to ``1.0.0b`` and repeat.

15. Update version numbers to final format

    In ``nnmt/__init__.py``, ``docs/source/conf.py``, and ``setup.py``,

    and commit

    .. code::

        git add nnmt/__init__.py docs/source/conf.py setup.py
        git commit -m 'Final update of version numbers'

16. Merge into ``master`` and ``develop`` and delete release branch

    .. code::

        git checkout master
        git pull
        git merge --no-ff release-<release_number>
        git push

        git checkout develop
        git pull
        git merge --no-ff release-<release_number>
        git push

        git checkout master
        git branch -d release-<release_number>

17. Create sdist and wheel

    .. code::

        conda activate temp0
        rm -r dist/
        python setup.py check
        python setup.py sdist bdist_wheel

18. Upload to test.pypi

    .. code::

        twine upload --repository-url https://test.pypi.org/legacy/ dist/*

19. Create new conda env and test install from test.pypi

    .. code::

        conda create -n temp2
        conda activate temp2
        conda install pip
        pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple nnmt==<release_number>
        cd tests/
        pytest
        cd ../

20. Upload to pypi

    Finally, you can upload your package to pypi.

    **WARNING**: This cannot be reversed, and the code cannot be changed
    afterwards, so the package needs to be in a final state.

    .. code::

        conda activate temp0
        twine upload dist/*

21. Test pip install

    .. code:

        conda create -n temp3
        conda activate temp3
        conda install pip
        pip install nnmt==<release_number>
        cd tests/
        pytest
        cd ../

22. Remove temporary conda environments

    .. code::

        conda activate base
        conda env remove -n temp0
        conda env remove -n temp1
        conda env remove -n temp2
        conda env remove -n temp3

23. Create Release on GitHub

    Optional: Create tag from command line

    .. code::

        git tag -a v<release_number>
        git push origin --tags

    Note that releases are a GitHub feature and can only be done on GitHub itself.

24. Upload compressed compy of the repository to Zenodo

25. Check readthedocs
