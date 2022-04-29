import React from 'react';
import renderer from 'react-test-renderer';

import App from './App';

describe('<App />', () => {
  it('has 3 children', () => {
    const tree = renderer.create(<App />).toJSON();
    expect(tree.children.length).toBe(3);
  });
});

it('renders correctly', () => {
    const tree = renderer.create(<App />).toJSON();
    expect(tree).toMatchSnapshot();
    expect(tree).toBeTruthy();
  });