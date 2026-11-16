# Symmetric-line mathematical tests

Date: 2026-11-15. Scope: research prototype `code/line_response.py` only.
No training/validation/test source photographs or GT files were opened in this audit.
All test images, strength fields and component masks are synthetic numerical fixtures;
none are evaluation data or evidence of real-road quality.

- **10/10 focused unittest tests passed**, 0.655 s.
- `python -m ruff check code/line_response.py tests/test_line_response.py --select E9,F`: passed.
- `python -m compileall -q code/line_response.py tests/test_line_response.py`: passed.

## Verified behavior

- Uniform levels 0, 1, 127, 128, 129, 253 and 255 yield exact zero after the final
  uint8 conversion, for widths 2 and 3. A straight 40-to-200 monotonic step yields
  no interior positive response in either axis orientation.
- A dark three-pixel line produces strong center evidence and no response on flat
  flanks. Ninety-degree rotations and mirror reflection preserve response within one
  uint8 level; a diagonal line is detected without claiming exact 45-degree isotropy.
- Offset sampling reflects the same border rather than wrapping the opposite side.
- A high-response long thin region survives mask filtering; a compact square, tiny
  isolated response, and disconnected weak component are rejected.
- Component gating retains the complete component when its evidence fraction passes,
  including the exact 10% boundary; background-only evidence cannot become foreground.
- Line/gate/augment configurations return boolean masks at original size for tiny,
  odd and extreme-aspect constant inputs.

## Mathematical nuance

The tangent three-pixel minimum **suppresses**, rather than guarantees elimination of,
isolated dots after Gaussian smoothing. A single dark pixel on a flat synthetic image
still produced response 11, versus 112 at the center of a dark three-pixel straight
line in the tested 180/35 intensity fixture. Later area/shape filtering remains needed.
This nuance was sent to the parent before reporting test success.

No formula implementation failure was found under the declared inputs. This report
does not claim real-image gains or justify admitting the prototype to production.
The existing full regression suite was not rerun during this restricted mathematical
audit because its earlier tests read bundled source photographs and GT fixtures.
