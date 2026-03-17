# spec05_structured_svg_atoms Structured Scenes Probe Report

Run: `spec05_structured_scenes_l3_d192_h384_ctx128_r2`

## Summary

- Cases: `26`
- Exact: `80.8%`
- Renderable: `92.3%`
- Materialized exact: `80.8%`
- SVG exact: `80.8%`

## Split Summary

| Split | Count | Exact | Renderable | SVG Exact |
| --- | ---: | ---: | ---: | ---: |
| `dev` | `8` | `75.0%` | `100.0%` | `75.0%` |
| `test` | `8` | `87.5%` | `100.0%` | `87.5%` |
| `train` | `10` | `80.0%` | `80.0%` | `80.0%` |

## Train Cases

### Train Single #1

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:single] [shape:circle] [color:blue] [size:big] [bg:none] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:none] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:paper] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:blue] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:none] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:paper] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:blue] [stroke:black] [sw:2] [/svg]
```

### Train Single #2

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:single] [shape:circle] [color:blue] [size:big] [bg:paper] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:blue] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:blue] [stroke:black] [sw:2] [/svg]
```

### Train Pair-H #3

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-h] [shape:circle] [shape2:rect] [color:blue] [color2:green] [size:big] [bg:paper] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-h] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:34] [cy:64] [r:16] [fill:blue] [stroke:black] [sw:2] [rect] [x:84] [y:56] [width:20] [height:16] [rx:7] [fill:green] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-h] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:34] [cy:64] [r:16] [fill:blue] [stroke:black] [sw:2] [rect] [x:84] [y:56] [width:20] [height:16] [rx:7] [fill:green] [stroke:black] [sw:2] [/svg]
```

### Train Pair-H #4

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-h] [shape:circle] [shape2:rect] [color:blue] [color2:green] [size:big] [bg:mint] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:pair-h] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:34] [cy:64] [r:16] [fill:blue] [stroke:black] [sw:2] [rect] [x:84] [y:56] [width:20] [height:16] [rx:7] [fill:green] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:pair-h] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:34] [cy:64] [r:16] [fill:blue] [stroke:black] [sw:2] [rect] [x:84] [y:56] [width:20] [height:16] [rx:7] [fill:green] [stroke:black] [sw:2] [/svg]
```

### Train Pair-V #5

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-v] [shape:circle] [shape2:triangle] [color:blue] [color2:green] [size:big] [bg:paper] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:blue] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:green] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:blue] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:green] [stroke:black] [sw:2] [/svg]
```

### Train Pair-V #6

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-v] [shape:circle] [shape2:triangle] [color:blue] [color2:green] [size:big] [bg:mint] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:blue] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:green] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:blue] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:green] [stroke:black] [sw:2] [/svg]
```

### Train Label #7

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:label-card] [color:blue] [bg:paper] [label:ai] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:label-card] [frame:card] [density:airy] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:white] [stroke:black] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:black] [sw:2] [text] [tx:64] [ty:64] [font:13] [anchor:middle] [fill:black] AI [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:label-card] [frame:card] [density:airy] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:white] [stroke:black] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:black] [sw:2] [text] [tx:64] [ty:64] [font:13] [anchor:middle] [fill:black] AI [/text] [/svg]
```

### Train Label #8

- Exact: `no `
- Renderable: `no `
- Valid SVG: `no `

Prompt:
```text
[task:svg] [layout:label-card] [color:blue] [bg:paper] [label:ai] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:label-card] [frame:card] [density:compact] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:white] [stroke:black] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:black] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:black] AI [/text] [/svg]
```

Response:
```text
[w:128] [h:128] [bg:paper] [bg:paper] [frame:card] [density:compact] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:white] [stroke:black] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:black] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:black] AI [/text] [/svg]
```

### Train Badge #9

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:badge] [shape:circle] [color:blue] [bg:paper] [label:ai] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:airy] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [circle] [cx:32] [cy:57] [r:11] [fill:blue] [stroke:black] [sw:2] [text] [tx:52] [ty:60] [font:11] [anchor:start] [fill:black] AI [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:airy] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [circle] [cx:32] [cy:57] [r:11] [fill:blue] [stroke:black] [sw:2] [text] [tx:52] [ty:60] [font:11] [anchor:start] [fill:black] AI [/text] [/svg]
```

### Train Badge #10

- Exact: `no `
- Renderable: `no `
- Valid SVG: `no `

Prompt:
```text
[task:svg] [layout:badge] [shape:circle] [color:blue] [bg:paper] [label:ai] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:compact] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [circle] [cx:28] [cy:58] [r:13] [fill:blue] [stroke:black] [sw:2] [text] [tx:48] [ty:61] [font:12] [anchor:start] [fill:black] AI [/text] [/svg]
```

Response:
```text
[w:128] [h:128] [bg:paper] [bg:paper] [frame:card] [density:compact] [x:10] [x:10] [width:108] [width:108] [height:76] [rx:18] [fill:white] [stroke:black] [fill:white] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:black] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:black] AI [/text] [/svg]
```

## Dev Cases

### Dev Single #1

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:single] [shape:circle] [color:gold] [size:big] [bg:mint] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:gold] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:gold] [stroke:black] [sw:2] [/svg]
```

### Dev Single #2

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:single] [shape:circle] [color:gold] [size:big] [bg:mint] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:single] [frame:card] [density:compact] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:32] [fill:gold] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:single] [frame:card] [density:compact] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:32] [fill:gold] [stroke:black] [sw:2] [/svg]
```

### Dev Pair-V #3

- Exact: `no `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-v] [shape:circle] [shape2:triangle] [color:gold] [color2:blue] [size:big] [bg:paper] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:gold] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:gold] [stroke:black] [sw:2] [/svg]
```

### Dev Pair-V #4

- Exact: `no `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-v] [shape:circle] [shape2:triangle] [color:gold] [color2:blue] [size:big] [bg:slate] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:gray] [stroke:white] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:gold] [stroke:white] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:white] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:gray] [stroke:white] [sw:2] [circle] [x:54] [cy:34] [r:16] [points:64,10|54,58|74,58] [stroke:white] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:white] [sw:2] [/svg]
```

### Dev Label #5

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:label-card] [color:blue] [bg:slate] [label:flow] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:airy] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:64] [font:13] [anchor:middle] [fill:white] FLOW [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:airy] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:64] [font:13] [anchor:middle] [fill:white] FLOW [/text] [/svg]
```

### Dev Label #6

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:label-card] [color:blue] [bg:slate] [label:ops] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:airy] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:64] [font:13] [anchor:middle] [fill:white] OPS [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:airy] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:64] [font:13] [anchor:middle] [fill:white] OPS [/text] [/svg]
```

### Dev Badge #7

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:badge] [shape:triangle] [color:blue] [bg:paper] [label:map] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:airy] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:31,45|19,67|43,67] [fill:blue] [stroke:black] [sw:2] [text] [tx:52] [ty:60] [font:11] [anchor:start] [fill:black] MAP [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:airy] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:31,45|19,67|43,67] [fill:blue] [stroke:black] [sw:2] [text] [tx:52] [ty:60] [font:11] [anchor:start] [fill:black] MAP [/text] [/svg]
```

### Dev Badge #8

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:badge] [shape:triangle] [color:blue] [bg:paper] [label:map] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:compact] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:28,44|16,70|40,70] [fill:blue] [stroke:black] [sw:2] [text] [tx:48] [ty:61] [font:12] [anchor:start] [fill:black] MAP [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:badge] [frame:card] [density:compact] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:28,44|16,70|40,70] [fill:blue] [stroke:black] [sw:2] [text] [tx:48] [ty:61] [font:12] [anchor:start] [fill:black] MAP [/text] [/svg]
```

## Test Cases

### Test Single #1

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:single] [shape:circle] [color:gold] [size:big] [bg:slate] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:gray] [stroke:white] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:gold] [stroke:white] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:single] [frame:card] [density:airy] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:gray] [stroke:white] [sw:2] [circle] [cx:64] [cy:64] [r:26] [fill:gold] [stroke:white] [sw:2] [/svg]
```

### Test Single #2

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:single] [shape:circle] [color:gold] [size:big] [bg:slate] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:single] [frame:card] [density:compact] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:gray] [stroke:white] [sw:2] [circle] [cx:64] [cy:64] [r:32] [fill:gold] [stroke:white] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:single] [frame:card] [density:compact] [rect] [x:14] [y:14] [width:100] [height:100] [rx:18] [fill:gray] [stroke:white] [sw:2] [circle] [cx:64] [cy:64] [r:32] [fill:gold] [stroke:white] [sw:2] [/svg]
```

### Test Pair-V #3

- Exact: `no `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-v] [shape:circle] [shape2:triangle] [color:gold] [color2:blue] [size:big] [bg:mint] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [cx:64] [cy:34] [r:16] [fill:gold] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:pair-v] [frame:card] [density:airy] [rect] [x:10] [y:20] [width:108] [height:88] [rx:18] [fill:white] [stroke:black] [sw:2] [circle] [x:54] [cy:34] [r:16] [x:54] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:black] [sw:2] [/svg]
```

### Test Pair-V #4

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:pair-v] [shape:circle] [shape2:triangle] [color:gold] [color2:blue] [size:big] [bg:paper] [frame:plain] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-v] [frame:plain] [density:airy] [circle] [cx:64] [cy:34] [r:16] [fill:gold] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:black] [sw:2] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:paper] [layout:pair-v] [frame:plain] [density:airy] [circle] [cx:64] [cy:34] [r:16] [fill:gold] [stroke:black] [sw:2] [polygon] [points:64,70|54,118|74,118] [fill:blue] [stroke:black] [sw:2] [/svg]
```

### Test Label #5

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:label-card] [color:blue] [bg:slate] [label:flow] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:compact] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:white] FLOW [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:compact] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:white] FLOW [/text] [/svg]
```

### Test Label #6

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:label-card] [color:blue] [bg:slate] [label:ops] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:compact] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:white] OPS [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:slate] [layout:label-card] [frame:card] [density:compact] [rect] [x:10] [y:26] [width:108] [height:76] [rx:18] [fill:gray] [stroke:white] [sw:2] [rect] [x:20] [y:42] [width:88] [height:36] [rx:10] [fill:blue] [stroke:white] [sw:2] [text] [tx:64] [ty:65] [font:14] [anchor:middle] [fill:white] OPS [/text] [/svg]
```

### Test Badge #7

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:badge] [shape:triangle] [color:blue] [bg:mint] [label:map] [frame:card] [density:airy] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:badge] [frame:card] [density:airy] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:31,45|19,67|43,67] [fill:blue] [stroke:black] [sw:2] [text] [tx:52] [ty:60] [font:11] [anchor:start] [fill:black] MAP [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:badge] [frame:card] [density:airy] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:31,45|19,67|43,67] [fill:blue] [stroke:black] [sw:2] [text] [tx:52] [ty:60] [font:11] [anchor:start] [fill:black] MAP [/text] [/svg]
```

### Test Badge #8

- Exact: `yes `
- Renderable: `yes `
- Valid SVG: `yes `

Prompt:
```text
[task:svg] [layout:badge] [shape:triangle] [color:blue] [bg:mint] [label:map] [frame:card] [density:compact] [OUT]
```

Expected:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:badge] [frame:card] [density:compact] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:28,44|16,70|40,70] [fill:blue] [stroke:black] [sw:2] [text] [tx:48] [ty:61] [font:12] [anchor:start] [fill:black] MAP [/text] [/svg]
```

Response:
```text
[svg] [w:128] [h:128] [bg:mint] [layout:badge] [frame:card] [density:compact] [rect] [x:10] [y:32] [width:108] [height:50] [rx:16] [fill:white] [stroke:black] [sw:2] [rect] [x:18] [y:42] [width:92] [height:30] [rx:10] [fill:white] [stroke:black] [sw:2] [polygon] [points:28,44|16,70|40,70] [fill:blue] [stroke:black] [sw:2] [text] [tx:48] [ty:61] [font:12] [anchor:start] [fill:black] MAP [/text] [/svg]
```
