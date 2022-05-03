# VADER-Sentiment-Analysis

A powerful [Go (golang)](http://golang.org) re-implementation of the **VADER** (Valence Aware Dictionary and sEntiment Reasoner).

[![Codecov](https://codecov.io/gh/knuppe/vader/branch/main/graph/badge.svg)](https://codecov.io/gh/knuppe/vader)
[![Go](https://github.com/knuppe/vader/actions/workflows/go.yml/badge.svg)](https://github.com/knuppe/vader/actions/workflows/go.yml)
[![CodeQL](https://github.com/knuppe/vader/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/knuppe/vader/actions/workflows/codeql-analysis.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/knuppe/vader)](https://goreportcard.com/report/github.com/knuppe/vader)
[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


VADER is a lexicon and rule-based sentiment analysis tool that is *specifically attuned to sentiments expressed in social media*.

## Main differences from the original implementation.

* **NO hardcoded rules**, everything is configurable in the lexicon files, allowing the addition of new languages without changing the code.
* **Blazingly fast!** In my machine evaluates 170.000 sentences per second vs 26.000 using the original python implementation.
* Emojis have valence values.
* Possibility to add exceptions to the rules.
* Possibility to use regular expressions in the rules.
* Possibility to normalize the input string: diacritics removal *(accents removal)*, string replacement.

## Available Languages

* English
* Portuguese
* German - Lexicon adapted from [GerVADER](https://github.com/KarstenAMF/GerVADER).

## Benchmark
```
Running tool: /usr/local/go/bin/go test -benchmem -run=^$ -bench ^BenchmarkPolarityScores$ github.com/knuppe/vader

goos: linux
goarch: amd64
pkg: github.com/knuppe/vader
cpu: AMD Ryzen 9 5950X 16-Core Processor            
BenchmarkPolarityScores/PolarityScores_en-32         	  176991	      7195 ns/op	    2357 B/op	      62 allocs/op
BenchmarkPolarityScores/PolarityScores_pt-32         	  144518	      7143 ns/op	    3568 B/op	      68 allocs/op
PASS
ok  	github.com/knuppe/vader	3.486s
```

## Usage

1. Download the desired lexicon file(s).

   - [English](https://github.com/knuppe/vader/raw/main/lexicons/en/en.zip)
   - [Portuguese](https://github.com/knuppe/vader/raw/main/lexicons/pt/pt.zip)
   - [German](https://github.com/knuppe/vader/raw/main/lexicons/de/de.zip)

2. Download the module.  
   `go get -u github.com/knuppe/vader@latest`

3. Have fun ;)
   
Here is a minimal Go program that uses this package in order
to analyze a sentence.

```go
package main
import (
	"fmt"
	"github.com/knuppe/vader"
)

func main(){
	// you need to download de lexicon zip files.
	vader, err := NewVader("lexicons/en/en.zip")
	if err != nil {
		panic(err)
	}

	score := vader.PolarityScores("VADER is smart, handsome, and funny!")

	fmt.Printf(
		"neg: %.3f, neu: %.3f, pos: %.3f, compound: %.3f (%s)",
		score.Negative,
		score.Neutral,
		score.Positive,
		score.Compound,
		score.Sentiment(),
	)
	// neg: 0.000, neu: 0.248, pos: 0.752, compound: 0.844 (positive)
}
```
## Citation
> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

## License (MIT)

Copyright (c) 2022 Gustavo Knuppe

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
